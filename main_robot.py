#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from moveit_msgs.msg import DisplayTrajectory, MotionPlanRequest, WorkspaceParameters, Constraints
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, TransformStamped
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

from moveit_msgs.srv import GetPositionIK

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

from moveit_msgs.msg import PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive

import time

class UR10Controller(Node):
    def __init__(self):
        super().__init__('ur10_controller')
        
        # Create action client for MoveGroup
        self.move_group_client = ActionClient(self, MoveGroup, '/move_action')
        
        # Create publisher for trajectory visualization
        self.display_trajectory_publisher = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10)
        
        # Create publisher for joint trajectory execution
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10)
        
        # Create subscription for joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        # TF2 Buffer and Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.current_joint_state = None
        
        self.get_logger().info('UR10 Controller node initialized')

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def get_current_pose(self):
        while self.current_joint_state is None:
            rclpy.spin_once(self)
            time.sleep(0.5)
        try:
            # Get the transform from base_link to tool0
            transform = self.tf_buffer.lookup_transform('base', 'tool0', rclpy.time.Time())
            
            # Create a PoseStamped message
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'base'
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position.x = transform.transform.translation.x
            pose_stamped.pose.position.y = transform.transform.translation.y
            pose_stamped.pose.position.z = transform.transform.translation.z
            pose_stamped.pose.orientation = transform.transform.rotation

            self.get_logger().info(f'Current end effector pose: {pose_stamped.pose}')
            return pose_stamped.pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'TF2 error: {e}')
            return None

    def plan_movement(self, target_pose):
        goal_msg = MoveGroup.Goal()
        
        # Set up the motion plan request
        motion_request = MotionPlanRequest()
        motion_request.group_name = "ur_manipulator"  # This is the standard name for UR robots
        motion_request.num_planning_attempts = 10
        motion_request.allowed_planning_time = 5.0
        motion_request.max_velocity_scaling_factor = 0.1
        motion_request.max_acceleration_scaling_factor = 0.1

        # Set the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "base"
        goal_pose.pose = target_pose
        motion_request.goal_constraints = [self.create_pose_goal(goal_pose)]

        # Set up workspace parameters (optional, adjust as needed)
        workspace = WorkspaceParameters()
        workspace.header.frame_id = "base"
        workspace.min_corner.x = -1.0
        workspace.min_corner.y = -1.0
        workspace.min_corner.z = -1.0
        workspace.max_corner.x = 1.0
        workspace.max_corner.y = 1.0
        workspace.max_corner.z = 1.0
        motion_request.workspace_parameters = workspace

        goal_msg.request = motion_request

        self.move_group_client.wait_for_server()
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

        result = result_future.result().result
        self.get_logger().info('Movement planning completed')
        
        return result

    def create_pose_goal(self, pose_stamped):
        constraints = Constraints()
        constraints.position_constraints = []
        constraints.orientation_constraints = []
        constraints.joint_constraints = []
        constraints.name = "pose_goal"

        pose_goal = Constraints()
        pose_goal.position_constraints.append(self.create_position_constraint(pose_stamped))
        pose_goal.orientation_constraints.append(self.create_orientation_constraint(pose_stamped))

        return pose_goal

    def create_position_constraint(self, pose_stamped):
        constraint = PositionConstraint()
        constraint.header = pose_stamped.header
        constraint.link_name = "tool0"  # This is the standard name for UR end effector
        constraint.target_point_offset.x = 0.0
        constraint.target_point_offset.y = 0.0
        constraint.target_point_offset.z = 0.0
        bounding_volume = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.001]  # 1mm tolerance
        bounding_volume.primitives.append(sphere)
        bounding_volume.primitive_poses.append(pose_stamped.pose)
        constraint.constraint_region = bounding_volume
        constraint.weight = 1.0

        return constraint

    def create_orientation_constraint(self, pose_stamped):
        constraint = OrientationConstraint()
        constraint.header = pose_stamped.header
        constraint.orientation = pose_stamped.pose.orientation
        constraint.link_name = "tool0"  # This is the standard name for UR end effector
        constraint.absolute_x_axis_tolerance = 0.1
        constraint.absolute_y_axis_tolerance = 0.1
        constraint.absolute_z_axis_tolerance = 0.1
        constraint.weight = 1.0

        return constraint

    def visualize_trajectory(self, trajectory):
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory.append(trajectory)
        self.display_trajectory_publisher.publish(display_trajectory)
        self.get_logger().info('Trajectory visualized in RViz')

    def execute_movement(self, trajectory):
        joint_trajectory = trajectory.joint_trajectory
        self.trajectory_publisher.publish(joint_trajectory)
        self.get_logger().info('Trajectory sent for execution')

def main(args=None):
    rclpy.init(args=args)
    controller = UR10Controller()

    try:
        # Wait for the first joint state message
        while controller.current_joint_state is None:
            rclpy.spin_once(controller)
        
        # Get current pose
        current_pose = controller.get_current_pose()
        if current_pose:
            controller.get_logger().info(f'Current end effector pose: {current_pose}')
        else:
            controller.get_logger().error('Failed to get current pose')
            return

        # Define target pose (you might want to adjust this based on the current pose)
        target_pose = Pose()
        target_pose.position = Point(x=current_pose.position.x + 0.1, y=current_pose.position.y, z=current_pose.position.z)
        target_pose.orientation = current_pose.orientation

        # Plan movement
        #planned_trajectory = controller.plan_movement(target_pose)
        planned_trajectory = False

        if planned_trajectory:
            # Visualize trajectory
            controller.visualize_trajectory(planned_trajectory.planned_trajectory[0])

            # Execute movement
            #controller.execute_movement(planned_trajectory.planned_trajectory[0])
        else:
            controller.get_logger().error('Failed to plan movement')

    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()