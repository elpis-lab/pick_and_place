#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from moveit_msgs.msg import DisplayTrajectory, MotionPlanRequest, WorkspaceParameters, Constraints
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, TransformStamped, Vector3
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from action_msgs.msg import GoalStatus
from moveit_msgs.srv import GetPositionIK

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from moveit_msgs.msg import PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive

import numpy as np
import cv2
import pyrealsense2 as rs
import requests
import json
from transforms3d.quaternions import mat2quat, quat2mat
import traceback

class UR10Controller(Node):
    def __init__(self):
        super().__init__('ur10_controller')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        self.current_joint_state = None
        self.current_pose = None

    def joint_state_callback(self, msg):
        self.current_joint_state = msg
        self.update_current_pose()

    def update_current_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())
            
            self.current_pose = Pose()
            self.current_pose.position.x = float(transform.transform.translation.x)
            self.current_pose.position.y = float(transform.transform.translation.y)
            self.current_pose.position.z = float(transform.transform.translation.z)
            self.current_pose.orientation = transform.transform.rotation

            self.get_logger().debug(f'Current end effector pose updated: {self.current_pose}')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF2 error in update_current_pose: {e}')

    def get_current_pose(self):
        return self.current_pose

class IntegratedGraspingSystem(Node):
    def __init__(self):
        super().__init__('integrated_grasping_system')
        
        self.ur10_controller = UR10Controller()
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = self.pipeline.start(self.config)
        
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        depth_intrinsics = depth_profile.get_intrinsics()
        self.fx = depth_intrinsics.fx
        self.fy = depth_intrinsics.fy
        self.cx = depth_intrinsics.ppx
        self.cy = depth_intrinsics.ppy
        
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        
        self.langsam_api_url = "http://localhost:8004/segment"
        self.anygrasp_api_url = "http://localhost:8001/get_grasp"
        self.prompt = "object"  # Default prompt, can be changed

        # Transformation matrix from end-effector to camera
        # This should be calibrated for your specific setup
        self.T_end_effector_camera = np.eye(4)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        return color_frame, depth_frame

    def plan_movement(self, target_pose):
        self.move_group_client.wait_for_server()

        goal_msg = MoveGroup.Goal()
        
        motion_request = MotionPlanRequest()
        motion_request.group_name = "ur_manipulator"
        motion_request.num_planning_attempts = 10
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = 0.1
        motion_request.max_acceleration_scaling_factor = 0.1

        goal_constraints = Constraints()
        goal_constraints.position_constraints = [self.create_position_constraint(target_pose)]
        goal_constraints.orientation_constraints = [self.create_orientation_constraint(target_pose)]
        motion_request.goal_constraints = [goal_constraints]

        goal_msg.request = motion_request

        self.get_logger().info('Sending goal request...')
        send_goal_future = self.move_group_client.send_goal_async(goal_msg)
        
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return None

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        return result_future.result().result

    def create_position_constraint(self, pose):
        constraint = PositionConstraint()
        constraint.header.frame_id = "base_link"
        constraint.link_name = "tool0"
        constraint.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        bounding_volume = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.001]  # 1mm tolerance
        bounding_volume.primitives.append(sphere)
        bounding_volume.primitive_poses.append(pose)
        constraint.constraint_region = bounding_volume
        constraint.weight = 1.0
        return constraint

    def create_orientation_constraint(self, pose):
        constraint = OrientationConstraint()
        constraint.header.frame_id = "base_link"
        constraint.orientation = pose.orientation
        constraint.link_name = "tool0"
        constraint.absolute_x_axis_tolerance = 0.1
        constraint.absolute_y_axis_tolerance = 0.1
        constraint.absolute_z_axis_tolerance = 0.1
        constraint.weight = 1.0
        return constraint

    def process_langsam(self, color_image):
        _, img_encoded = cv2.imencode('.jpg', color_image)
        files = {'rgb_image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'prompt': self.prompt}
        response = requests.post(self.langsam_api_url, files=files, data=data)
        return response.json()

    def process_anygrasp(self, color_image, depth_image):
        _, color_encoded = cv2.imencode('.jpg', color_image)
        _, depth_encoded = cv2.imencode('.png', depth_image)
        files = {
            'color_image': ('color.jpg', color_encoded.tobytes(), 'image/jpeg'),
            'depth_image': ('depth.png', depth_encoded.tobytes(), 'image/png')
        }
        response = requests.post(self.anygrasp_api_url, files=files)
        return response.json()

    def project_3d_to_2d(self, point_3d):
        x, y, z = point_3d
        u = (x * self.fx / z) + self.cx
        v = (y * self.fy / z) + self.cy
        return np.array([u, v])

    def fuse_grasps(self, langsam_result, anygrasp_result):
        if not langsam_result['results'] or 'grasps' not in anygrasp_result:
            return None

        segmentation_center = np.array(langsam_result['results'][0]['center'])
        grasps = anygrasp_result['grasps']

        # Project 3D grasp points to 2D
        projected_grasps = [self.project_3d_to_2d(g['translation']) for g in grasps]

        # Find the grasp closest to the segmentation center
        closest_grasp = min(zip(grasps, projected_grasps), 
                            key=lambda x: np.linalg.norm(x[1] - segmentation_center))
        return closest_grasp[0]

    def calculate_final_pose(self, grasp, robot_pose):
        # T_base to end_effector (from robot_pose)
        T_baseToEndEffector = np.eye(4)
        T_baseToEndEffector[:3, 3] = [float(robot_pose.position.x), float(robot_pose.position.y), float(robot_pose.position.z)]
        q = [float(robot_pose.orientation.x), float(robot_pose.orientation.y), float(robot_pose.orientation.z), float(robot_pose.orientation.w)]
        T_baseToEndEffector[:3, :3] = quat2mat(q)

        # T_camera to Grasping object (from grasp)
        T_cameraToGrasp = np.eye(4)
        T_cameraToGrasp[:3, 3] = [float(x) for x in grasp['translation']]
        T_cameraToGrasp[:3, :3] = np.array([[float(x) for x in row] for row in grasp['rotation_matrix']])

        # Calculate final transformation
        T_final = T_baseToEndEffector @ self.T_end_effector_camera @ T_cameraToGrasp

        # Extract position and orientation
        position = T_final[:3, 3]
        orientation = mat2quat(T_final[:3, :3])

        final_pose = Pose()
        final_pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        final_pose.orientation = Quaternion(x=float(orientation[1]), y=float(orientation[2]), z=float(orientation[3]), w=float(orientation[0]))

        return final_pose

    def demo_pick_place(self):
        self.get_logger().info("Starting pick and place demo")
        
        # 1. Move to a pre-grasp position
        pre_grasp_pose = Pose(
            position=Point(x=0.4, y=0.0, z=0.5),
            orientation=Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        )

        # result = self.plan_movement(pre_grasp_pose)
        #if not result:
        #    self.get_logger().error("Failed to plan movement to pre-grasp position")
        #    return
        
        # 2. Get current pose and plan grasp
        current_pose = self.ur10_controller.get_current_pose()
        if not current_pose:
            self.get_logger().error("Failed to get current pose")
            return
        
        color_frame, depth_frame = self.get_frames()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        langsam_result = self.process_langsam(color_image)
        anygrasp_result = self.process_anygrasp(color_image, depth_image)
        
        fused_grasp = self.fuse_grasps(langsam_result, anygrasp_result)
        if not fused_grasp:
            self.get_logger().error("Failed to find a suitable grasp")
            return
        
        grasp_pose = self.calculate_final_pose(fused_grasp, current_pose)
        
        print("Grasp pose:", grasp_pose)
        # 3. Move to grasp pose
        # result = self.plan_movement(grasp_pose)
        # if not result:
        #     self.get_logger().error("Failed to plan movement to grasp pose")
        #     return
        
        # # 4. Move to place position
        # place_pose = Pose(
        #     position=Point(x=0.4, y=0.4, z=0.3),
        #     orientation=Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        # )
        # result = self.plan_movement(place_pose)
        # if not result:
        #     self.get_logger().error("Failed to plan movement to place position")
        #     return
        
        # # 5. Move back to initial position
        # result = self.plan_movement(pre_grasp_pose)
        # if not result:
        #     self.get_logger().error("Failed to plan movement back to initial position")
        #     return
        
        self.get_logger().info("Pick and place demo completed successfully")

    def run(self):
        while rclpy.ok():
            try:
                self.demo_pick_place()
                rclpy.spin_once(self, timeout_sec=0.1)
            except Exception as e:
                self.get_logger().error(f'Unexpected error in demo_pick_place: {e}')
                self.get_logger().error(f'Error type: {type(e).__name__}')
                self.get_logger().error(f'Traceback: {traceback.format_exc()}')

def main(args=None):
    rclpy.init(args=args)
    integrated_system = IntegratedGraspingSystem()
    try:
        integrated_system.run()
    except KeyboardInterrupt:
        pass
    finally:
        integrated_system.pipeline.stop()
        integrated_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()