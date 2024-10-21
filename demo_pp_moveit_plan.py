import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
import numpy as np
import cv2
import requests
import json
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from transforms3d.quaternions import mat2quat, quat2mat
import open3d as o3d
import sys
import time
from moveit_msgs.msg import DisplayTrajectory
from moveit_msgs.srv import GetPositionIK
from moveit_commander import MoveGroupCommander, RobotCommander

class IntegratedGraspingSystem(Node):
    def __init__(self, visualize_o3d=False, visualize_cv2=True):
        super().__init__('integrated_grasping_system')
        self.pipeline = None
        self.align = None
        self.langsam_api_base = "http://localhost:8004"
        self.anygrasp_api_base = "http://localhost:8001"
        self.langsam_api_url = "http://localhost:8004/segment"
        self.anygrasp_api_url = "http://localhost:8001/get_grasp"
        self.prompt = "banana"  # Initial prompt, can be changed
        self.visualize_o3d = visualize_o3d
        self.visualize_cv2 = visualize_cv2
        self.t_end_affector_camera = np.eye(4)  # Replace with actual transformation
        self.vis = None
        if self.visualize_o3d:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()
        
        # RealSense parameters
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.depth_scale = None

        # RViz visualization
        self.marker_publisher = self.create_publisher(Marker, 'grasp_pose', 10)
        self.trajectory_publisher = self.create_publisher(DisplayTrajectory, 'move_group/display_planned_path', 10)

        # MoveIt setup
        self.robot = RobotCommander()
        self.move_group = MoveGroupCommander("arm")  # Replace "arm" with your robot's arm group name
        self.ik_service = self.create_client(GetPositionIK, 'compute_ik')

    def setup_realsense(self, json_file):
        pipeline = rs.pipeline()
        config = rs.config()
        profile = pipeline.start(config)
        device = profile.get_device()
        adv_mode = rs.rs400_advanced_mode(device)
        
        if not adv_mode.is_enabled():
            adv_mode.toggle_advanced_mode(True)
            print("Advanced mode enabled.")

        with open(json_file, 'r') as f:
            json_text = f.read().strip()
            adv_mode.load_json(json_text)

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale is: {self.depth_scale} meters per unit")

        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics_depth = depth_stream.as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intrinsics_depth.fx, intrinsics_depth.fy
        self.cx, self.cy = intrinsics_depth.ppx, intrinsics_depth.ppy

        self.pipeline = pipeline
        self.align = rs.align(rs.stream.color)

    def check_apis(self):
        apis = [
            ("LangSAM", f"{self.langsam_api_base}/"),
            ("AnyGrasp", f"{self.anygrasp_api_base}/")
        ]
        
        all_up = True
        for name, url in apis:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    self.get_logger().info(f"{name} API is up and running.")
                else:
                    self.get_logger().error(f"{name} API returned status code {response.status_code}")
                    all_up = False
            except requests.ConnectionError:
                self.get_logger().error(f"Failed to connect to {name} API at {url}")
                all_up = False
        
        if not all_up:
            self.get_logger().error("One or more required APIs are not accessible. Exiting.")
            sys.exit(1)

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return color_frame, depth_frame

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
        u = int((x * self.fx + self.cx) / z)
        v = int((y * self.fy + self.cy) / z)
        return np.array([u, v])

    def fuse_grasps(self, langsam_result, anygrasp_result):
        if not langsam_result['results'] or 'grasps' not in anygrasp_result:
            return None

        segmentation_center = np.array(langsam_result['results'][0]['center'])
        grasps = anygrasp_result['grasps']

        projected_grasps = [self.project_3d_to_2d(g['translation']) for g in grasps]
        closest_grasp = min(zip(grasps, projected_grasps), 
                            key=lambda x: np.linalg.norm(x[1] - segmentation_center))
        return closest_grasp[0]

    def visualize_cv2_out(self, color_image, langsam_result, anygrasp_result, fused_grasp):
        vis_image = color_image.copy()

        if langsam_result and 'results' in langsam_result and langsam_result['results']:
            mask = np.array(langsam_result['results'][0]['mask'])
            vis_image = cv2.addWeighted(vis_image, 1, cv2.applyColorMap(mask.astype(np.uint8) * 100, cv2.COLORMAP_JET), 0.5, 0)

        if anygrasp_result and 'grasps' in anygrasp_result:
            for grasp in anygrasp_result['grasps']:
                u, v = self.project_3d_to_2d(grasp['translation'])
                cv2.circle(vis_image, (u, v), 5, (0, 255, 0), -1)

        if fused_grasp:
            u, v = self.project_3d_to_2d(fused_grasp['translation'])
            cv2.circle(vis_image, (u, v), 10, (255, 0, 0), -1)
            
            gripper_size = 30
            cv2.line(vis_image, (u-gripper_size, v), (u+gripper_size, v), (255, 0, 0), 2)
            cv2.line(vis_image, (u, v-gripper_size), (u, v+gripper_size), (255, 0, 0), 2)

        cv2.putText(vis_image, f"Prompt: {self.prompt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Integrated Grasping System", vis_image)
        cv2.waitKey(1)

    def calculate_final_pose(self, grasp, T_endaffectortoCamera, robot_pose):
        T_baseToEndaffector = np.eye(4)
        T_baseToEndaffector[:3, 3] = [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
        q = [robot_pose.orientation.x, robot_pose.orientation.y, robot_pose.orientation.z, robot_pose.orientation.w]
        T_baseToEndaffector[:3, :3] = quat2mat(q)

        T_cameraToGrasp = np.eye(4)
        T_cameraToGrasp[:3, 3] = grasp['translation']
        T_cameraToGrasp[:3, :3] = np.array(grasp['rotation_matrix'])

        T_final = T_baseToEndaffector @ T_endaffectortoCamera @ T_cameraToGrasp

        position = T_final[:3, 3]
        orientation = mat2quat(T_final[:3, :3])

        final_pose = Pose()
        final_pose.position = Point(x=position[0], y=position[1], z=position[2])
        final_pose.orientation = Quaternion(x=orientation[1], y=orientation[2], z=orientation[3], w=orientation[0])

        return final_pose
    
    def plan_to_pose(self, target_pose):
        self.move_group.set_pose_target(target_pose)
        plan = self.move_group.plan()
        
        # Publish the trajectory for visualization in RViz
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan[1])
        self.trajectory_publisher.publish(display_trajectory)

        return plan
    
    def publish_grasp_marker(self, pose):
        marker = Marker()
        marker.header.frame_id = "base_link"  # Adjust if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "grasp_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.05  # Arrow width
        marker.scale.z = 0.05  # Arrow height
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red color
        marker.lifetime.sec = 0  # 0 means forever

        self.marker_publisher.publish(marker)

    def create_gripper_geometry(self, grasp, width=0.08, depth=0.05):
        left = o3d.geometry.TriangleMesh.create_box(width=depth, height=width/4, depth=width/2)
        right = o3d.geometry.TriangleMesh.create_box(width=depth, height=width/4, depth=width/2)
        left.translate((-depth/2, 0, width/4))
        right.translate((-depth/2, 0, -width/4 - width/2))
        gripper = left + right

        R = np.array(grasp['rotation_matrix'])
        t = np.array(grasp['translation'])
        gripper.rotate(R, center=(0, 0, 0))
        gripper.translate(t)

        return gripper

    def visualize_scene_o3d(self, color_image, depth_image, segmentation, grasps):
        color_o3d = o3d.geometry.Image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, convert_rgb_to_intensity=False)

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            color_image.shape[1], color_image.shape[0], self.fx, self.fy, self.cx, self.cy)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)

        self.vis.clear_geometries()
        self.vis.add_geometry(pcd)

        if segmentation and segmentation['results']:
            center = segmentation['results'][0]['center']
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(np.array(center + [0.5]))  # Adding depth estimate
            sphere.paint_uniform_color([1, 0, 0])  # Red color
            self.vis.add_geometry(sphere)

        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red, Green, Blue
        for i, grasp in enumerate(grasps[:3]):  # Visualize top 3 grasps
            gripper = self.create_gripper_geometry(grasp)
            gripper.paint_uniform_color(colors[i])
            self.vis.add_geometry(gripper)

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            coord_frame.rotate(grasp['rotation_matrix'])
            coord_frame.translate(grasp['translation'])
            self.vis.add_geometry(coord_frame)

        self.vis.poll_events()
        self.vis.update_renderer()


    def run(self):
        self.check_apis()
        self.setup_realsense('realsense_config.json')
        
        while rclpy.ok():
            color_frame, depth_frame = self.get_frames()
            
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            langsam_result = self.process_langsam(color_image)
            anygrasp_result = self.process_anygrasp(color_image, depth_image)

            fused_grasp = self.fuse_grasps(langsam_result, anygrasp_result)

            if fused_grasp:
                # For demonstration, we'll just print the final pose
                # In a real system, you would send this to your robot controller
                robot_pose = Pose()  # Placeholder, replace with actual robot pose
                final_pose = self.calculate_final_pose(fused_grasp, self.t_end_affector_camera, robot_pose)
                print("Final pose:", final_pose)

                # Publish the grasp pose as a marker in RViz
                self.publish_grasp_marker(final_pose)

            if self.visualize_o3d:
                self.visualize_scene_o3d(color_image, depth_image, langsam_result, anygrasp_result['grasps'])

            if self.visualize_cv2:
                self.visualize_cv2_out(color_image, langsam_result, anygrasp_result, fused_grasp)

            rclpy.spin_once(self, timeout_sec=0.1)

        if self.visualize_o3d:
            self.vis.destroy_window()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    integrated_system = IntegratedGraspingSystem(visualize_o3d=False, visualize_cv2=False)
    try:
        integrated_system.run()
    except KeyboardInterrupt:
        pass
    finally:
        integrated_system.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()