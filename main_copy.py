#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from threading import Thread
import numpy as np
import cv2
import pyrealsense2 as rs
import requests
from transforms3d.quaternions import mat2quat, quat2mat
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from geometry_msgs.msg import Pose, Point, Quaternion
from robot_control import RobotControl  # Import from existing robot_control.py

class IntegratedPickPlace(Node):
    def __init__(self):
        super().__init__('integrated_pick_place')
        
        # Initialize RobotControl
        self.robot_control = RobotControl()
        
        # Initialize camera-related variables
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        
        # Setup RealSense camera
        self.setup_realsense("realsense_config.json")  # Make sure to provide the correct path to your JSON file
        
        # API endpoints
        self.langsam_api_url = "http://localhost:8004/segment"
        self.anygrasp_api_url = "http://localhost:8001/get_grasp"
        self.gripper_api_url = "http://localhost:8005"
        
        # Default segmentation prompt
        self.prompt = "banana"
        
        # Transform matrix from end-effector to camera (needs to be calibrated)
        self.temp_xyz = [0.033051, 0.0395279, -0.0370267]
        self.temp_xyzw = [0.0134823, 0.0504259, 0.997471, 0.04823]
        if self.temp_xyz and self.temp_xyzw:
            self.T_end_effector_camera = np.eye(4)
            self.T_end_effector_camera[:3, 3] = self.temp_xyz
            self.T_end_effector_camera[:3, :3] = quat2mat(self.temp_xyzw)
        else:
            self.T_end_effector_camera = np.eye(4)

    def setup_realsense(self, json_file):
        """Setup RealSense camera with advanced mode configuration"""
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable streams
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline
            profile = pipeline.start(config)
            device = profile.get_device()
            adv_mode = rs.rs400_advanced_mode(device)
            
            if not adv_mode.is_enabled():
                adv_mode.toggle_advanced_mode(True)
                print("Advanced mode enabled.")

            # Load JSON configuration
            with open(json_file, 'r') as f:
                json_text = f.read().strip()
                adv_mode.load_json(json_text)

            # Get depth scale
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth Scale is: {self.depth_scale} meters per unit")

            # Get camera intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            intrinsics_depth = depth_stream.as_video_stream_profile().get_intrinsics()
            self.fx, self.fy = intrinsics_depth.fx, intrinsics_depth.fy
            self.cx, self.cy = intrinsics_depth.ppx, intrinsics_depth.ppy

            self.pipeline = pipeline
            self.align = rs.align(rs.stream.color)
            
            print("RealSense camera setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up RealSense camera: {e}")
            traceback.print_exc()
            raise

    def get_frames(self):
        """Get aligned color and depth frames from RealSense camera"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image

    async def process_langsam(self, color_image):
        """Process image with Langsam segmentation API"""
        _, img_encoded = cv2.imencode('.jpg', color_image)
        files = {'rgb_image': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}
        data = {'prompt': self.prompt}
        
        try:
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: requests.post(self.langsam_api_url, files=files, data=data)
                )
            return response.json()
        except Exception as e:
            self.get_logger().error(f'Error calling Langsam API: {e}')
            return None

    async def process_anygrasp(self, color_image, depth_image):
        """Process images with AnyGrasp API"""
        _, color_encoded = cv2.imencode('.jpg', color_image)
        _, depth_encoded = cv2.imencode('.png', depth_image)
        files = {
            'color_image': ('color.jpg', color_encoded.tobytes(), 'image/jpeg'),
            'depth_image': ('depth.png', depth_encoded.tobytes(), 'image/png')
        }
        
        try:
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: requests.post(self.anygrasp_api_url, files=files)
                )
            return response.json()
        except Exception as e:
            self.get_logger().error(f'Error calling AnyGrasp API: {e}')
            return None

    async def control_gripper(self, action):
        """Control the gripper through its API"""
        try:
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: requests.post(f"{self.gripper_api_url}/{action}")
                )
            return response.json()
        except Exception as e:
            self.get_logger().error(f'Error controlling gripper: {e}')
            return None

    def project_3d_to_2d(self, point_3d):
        """Project 3D point to 2D image coordinates"""
        x, y, z = point_3d
        u = (x * self.fx / z) + self.cx
        v = (y * self.fy / z) + self.cy
        return np.array([u, v])

    def visualize_results(self, color_image, depth_image, langsam_result, anygrasp_result, fused_grasp=None):
        """Create visualization of processing results"""
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        langsam_viz = color_image.copy()
        grasp_viz = color_image.copy()
        combined_viz = color_image.copy()

        if langsam_result and 'results' in langsam_result:
            for result in langsam_result['results']:    
                x, y, w, h = result['bounding_box']
                cv2.rectangle(langsam_viz, (int(x), int(y)), 
                            (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.circle(langsam_viz, (int(result['center'][0]), int(result['center'][1])), 
                          5, (255, 0, 0), -1)

        if anygrasp_result and 'grasps' in anygrasp_result:
            for grasp in anygrasp_result['grasps']:
                point_3d = grasp['translation']
                point_2d = self.project_3d_to_2d(point_3d)
                cv2.circle(grasp_viz, (int(point_2d[0]), int(point_2d[1])), 
                          5, (0, 0, 255), -1)

        if fused_grasp:
            point_3d = fused_grasp['translation']
            point_2d = self.project_3d_to_2d(point_3d)
            cv2.circle(combined_viz, (int(point_2d[0]), int(point_2d[1])), 
                      10, (0, 255, 255), -1)

        top_row = np.hstack((color_image, depth_colormap))
        bottom_row = np.hstack((langsam_viz, grasp_viz))
        combined_image = np.vstack((top_row, bottom_row))

        cv2.imshow('Processing Results', combined_image)
        self.get_logger().info("Press 'ESC' to continue...")
        
        while True:
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
        
        cv2.destroyAllWindows()

    def fuse_grasps(self, langsam_result, anygrasp_result):
        """Fuse segmentation and grasp results"""
        if not langsam_result or not langsam_result.get('results') or 'grasps' not in anygrasp_result:
            return None

        segmentation_center = np.array(langsam_result['results'][0]['center'])
        grasps = anygrasp_result['grasps']

        projected_grasps = [self.project_3d_to_2d(g['translation']) for g in grasps]
        closest_grasp = min(zip(grasps, projected_grasps), 
                          key=lambda x: np.linalg.norm(x[1] - segmentation_center))
        return closest_grasp[0]

    def calculate_final_pose(self, grasp, robot_pose):
        """Calculate final robot pose from grasp and current robot pose"""
        T_baseToEndEffector = np.eye(4)
        T_baseToEndEffector[:3, 3] = [
            robot_pose.position.x,
            robot_pose.position.y,
            robot_pose.position.z
        ]
        q = [robot_pose.orientation.w,
             robot_pose.orientation.x,
             robot_pose.orientation.y,
             robot_pose.orientation.z]
        T_baseToEndEffector[:3, :3] = quat2mat(q)

        T_cameraToGrasp = np.eye(4)
        T_cameraToGrasp[:3, 3] = grasp['translation']
        T_cameraToGrasp[:3, :3] = np.array(grasp['rotation_matrix'])
        self.get_logger().info(f"Grasp translation: {grasp['translation']}")

        T_final = T_baseToEndEffector @ self.T_end_effector_camera @ T_cameraToGrasp

        position = T_final[:3, 3]
        orientation = mat2quat(T_final[:3, :3])

        final_pose = Pose()
        final_pose.position = Point(x=position[0], y=position[1], z=position[2])
        final_pose.orientation = Quaternion(
            w=orientation[0],
            x=orientation[1],
            y=orientation[2],
            z=orientation[3]
        )

        return final_pose

    async def execute_pick_place(self):
        """Execute the complete pick and place operation"""
        try:
            self.get_logger().info("Starting pick and place operation")
            
            # Give more time for the node to initialize and receive joint states
            for _ in range(10):  # Try up to 10 times
                await asyncio.sleep(0.1)  # Use asyncio.sleep instead of spin_once
                if self.robot_control.current_joint_state is not None:
                    break
            
            # Get current robot pose
            current_pose = self.robot_control.get_current_ee_pose()
            if not current_pose:
                self.get_logger().error("Failed to get current robot pose")
                return False
            
            # Get camera frames
            color_image, depth_image = self.get_frames()
            if color_image is None or depth_image is None:
                self.get_logger().error("Failed to get camera frames")
                return False
            
            # Show camera frames
            cv2.imshow("RealSense Color Image", color_image)
            cv2.imshow("RealSense Depth Image", cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
            cv2.waitKey(30)
            
            # Process with vision APIs
            langsam_result = await self.process_langsam(color_image)

            anygrasp_result = await self.process_anygrasp(color_image, depth_image)

            print("Langsam Result: ", langsam_result.keys())
            print("AnyGrasp Result: ", anygrasp_result.keys())
            
            # Fuse grasp results
            fused_grasp = self.fuse_grasps(langsam_result, anygrasp_result)
            if not fused_grasp:
                self.get_logger().error("Failed to find suitable grasp")
                return False
            
            # Show results and wait for confirmation
            self.visualize_results(color_image, depth_image, langsam_result, 
                                 anygrasp_result, fused_grasp)
            
            # Calculate target pose
            target_pose = self.calculate_final_pose(fused_grasp, current_pose)
            self.get_logger().info(f"Calculated target pose: {target_pose}")
            
            # Control robot
            await self.control_gripper("open")
            trajectory = self.robot_control.plan_to_pose(target_pose)
            if trajectory:
                execute_input = input("Do you want to execute the trajectory? (y/n): ")
                if execute_input.lower() == 'y':
                    if self.robot_control.execute_trajectory(trajectory):
                        await self.control_gripper("close")
                        self.get_logger().info("Pick operation completed")
                        return True
            
            self.get_logger().error("Failed to execute pick operation")
            return False
            
        except Exception as e:
            self.get_logger().error(f'Error in pick and place operation: {e}')
            self.get_logger().error(traceback.format_exc())
            return False

    def start_pick_place(self):
        """Non-async wrapper to start the pick and place operation"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_pick_place())

def main(args=None):
    rclpy.init(args=args)
    
    pick_place_system = IntegratedPickPlace()
    executor = MultiThreadedExecutor()
    executor.add_node(pick_place_system.robot_control)
    executor.add_node(pick_place_system)
    
    # Create a separate thread for the executor
    executor_thread = Thread(target=executor.spin)
    executor_thread.start()
    
    try:
        # Run the pick and place operation
        success = pick_place_system.start_pick_place()
        if success:
            print("Pick and place operation completed successfully")
        else:
            print("Pick and place operation failed")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    finally:
        # Cleanup
        executor.shutdown()
        pick_place_system.pipeline.stop()
        pick_place_system.destroy_node()
        rclpy.shutdown()
        executor_thread.join()

if __name__ == '__main__':
    main()