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
#from marker_helper import * 
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Vector3
from list_realsense import list_realsense_cameras

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

        # Add Gripper offset
        self.gripper_offset = 0.09

        # Get id of the realsense D435 (check for string in the name)
        cameras = list_realsense_cameras()
        print(f"Found {len(cameras)} RealSense camera(s):")
        for i, camera in enumerate(cameras, 1):
            if "D435" in camera['name']:
                self.realsense_id = i
                self.realsense_serial_no = camera['serial_number']  
                break
        
        # Setup RealSense camera
        self.setup_realsense("realsense_config.json")  # Make sure to provide the correct path to your JSON file
        
        # API endpoints
        self.langsam_api_url = "http://localhost:8004/segment"
        self.anygrasp_api_url = "http://localhost:8001/get_grasp"
        self.gripper_api_url = "http://localhost:8005"
        self.place_api_url = "http://localhost:8007/process"
    
        # Default segmentation prompt
        self.prompt = "banana" 
        self.place_prompt = "plate"
        # Transform matrix from end-effector to camera (needs to be calibrated)
        #self.temp_xyz = [0.033051, 0.0395279, -0.0370267]
        #self.temp_xyzw = [0.0134823, 0.0504259, 0.997471, 0.04823]

        self.temp_xyz = [0.0195301, 0.0514428, 0.0542184]
        self.temp_xyzw = [0.000546104,0.0378262,0.999282, -0.00209989]
        if self.temp_xyz and self.temp_xyzw:
            self.T_end_effector_camera = np.eye(4)
            self.T_end_effector_camera[:3, 3] = self.temp_xyz
            self.T_end_effector_camera[:3, :3] = quat2mat(self.temp_xyzw)
        else:
            self.T_end_effector_camera = np.eye(4)
        
        self.get_logger().info("Initializing Interactive Markers")
        self.setup_interactive_markers()

        # Adding switching between cartesian path planning and RRT planning
        self.cartesian_path = True

        

    def setup_realsense(self, json_file):
        """Setup RealSense camera with advanced mode configuration"""
        try:
            pipeline = rs.pipeline()
            config = rs.config()

            # To select a specific camera, add this line before enabling streams:
            print("Connecting to Intel D435 camera with ID: {} and serial number: {}".format(self.realsense_id, self.realsense_serial_no))
            config.enable_device(self.realsense_serial_no)  # Replace with your camera's serial number

            
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
            # Log Intrinsics
            self.get_logger().info("============================================")
            self.get_logger().info("Camera Intrinsics")
            self.get_logger().info(f"fx: {self.fx}, fy: {self.fy}")
            self.get_logger().info(f"cx: {self.cx}, cy: {self.cy}")
            self.get_logger().info("============================================")
            
            print("RealSense camera setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up RealSense camera: {e}")
            traceback.print_exc()
            raise

    def setup_interactive_markers(self):
        """Initialize interactive marker server"""
        self.marker_server = InteractiveMarkerServer(
            self,
            'pick_point_markers'
        )
        self.current_marker = None

    def create_pick_point_marker(self, target_pose):
        """Create an interactive marker at the target pick point"""
        try:
            # Create the interactive marker
            int_marker = InteractiveMarker()
            int_marker.header.frame_id = "base_link"
            int_marker.header.stamp = self.get_clock().now().to_msg()
            int_marker.name = "pick_point"
            int_marker.description = "Pick Point Target"
            int_marker.pose = target_pose
            int_marker.scale = 0.2

            # Create a marker for visualization
            marker = Marker()
            marker.type = Marker.ARROW
            marker.scale = Vector3(x=0.2, y=0.02, z=0.02)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Create marker control
            marker_control = InteractiveMarkerControl()
            marker_control.always_visible = True
            marker_control.markers.append(marker)
            int_marker.controls.append(marker_control)

            # Add 6-DOF controls
            for axis, is_rot in [
                ((1.0, 0.0, 0.0), True), ((0.0, 1.0, 0.0), True), ((0.0, 0.0, 1.0), True),
                ((1.0, 0.0, 0.0), False), ((0.0, 1.0, 0.0), False), ((0.0, 0.0, 1.0), False)
            ]:
                control = InteractiveMarkerControl()
                control.orientation.w = 1.0
                control.orientation.x = float(axis[0])
                control.orientation.y = float(axis[1])
                control.orientation.z = float(axis[2])
                control.name = f"{'rotate' if is_rot else 'move'}_{['x', 'y', 'z'][axis.index(1.0)]}"
                control.interaction_mode = (
                    InteractiveMarkerControl.ROTATE_AXIS if is_rot
                    else InteractiveMarkerControl.MOVE_AXIS
                )
                int_marker.controls.append(control)

            # Add the interactive marker to the server
            self.marker_server.insert(int_marker)
            self.marker_server.applyChanges()
            self.current_marker = int_marker
            
            self.get_logger().info("Interactive marker created successfully")
            
        except Exception as e:
            self.get_logger().error(f"Error creating marker: {e}")
            self.get_logger().error(traceback.format_exc())

    def marker_feedback_callback(self, feedback):
        """Handle interactive marker feedback"""
        if hasattr(feedback, 'event_type'):  # Check if feedback has event_type attribute
            self.get_logger().info(f"Received feedback from marker: {feedback.marker_name}")
            self.get_logger().info(f"New pose: {feedback.pose}")


    def update_marker_pose(self, new_pose):
        """Update the position of the interactive marker"""
        if self.current_marker:
            self.current_marker.pose = new_pose
            self.marker_server.insert(self.current_marker)
            self.marker_server.applyChanges()

    def remove_marker(self):
        """Remove the interactive marker"""
        if self.current_marker:
            self.marker_server.erase(self.current_marker.name)
            self.marker_server.applyChanges()
            self.current_marker = None

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

        top_row = np.hstack((color_image, combined_viz)) #depth_image
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
        self.get_logger().info(f"All Grasps: {closest_grasp}")
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
    
    async def process_place_api(self):
        """
        Asynchronously call the place API endpoint
        
        Returns:
            Optional[Dict]: The API response as dictionary, or None if request fails
        """
        data = {'prompt': self.place_prompt}
        
        try:
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: requests.post(
                        self.place_api_url,
                        json=data,
                        headers={'Content-Type': 'application/json'}
                    )
                )
                
            return response.json()
        except Exception as e:
            self.logger.error(f'Error calling Place API: {e}')
            return None

    async def execute_pick_place(self):
        """Execute the complete pick and place operation"""
        try:
            langsam_result_place = await self.process_place_api()
            self.get_logger().info("Starting pick and place operation")
            buffer_pose = None
            # Give more time for the node to initialize and receive joint states
            for _ in range(10):  # Try up to 10 times
                await asyncio.sleep(0.1)  # Use asyncio.sleep instead of spin_once
                if self.robot_control.current_joint_state is not None:
                    break
            
            # Get current robot pose
            self.get_logger().info("============================================")
            self.get_logger().info("Getting current robot pose - End Effector")

            current_pose = self.robot_control.get_current_ee_pose()
            # Get current base pose
            self.get_logger().info("============================================")
            self.get_logger().info("Getting current robot pose - Base Link")
            current_base_pose = self.robot_control.get_current_base_pose()
            self.get_logger().info(f"============================================")
            if not current_pose:
                self.get_logger().error("Failed to get current robot pose")
                return False
            
            buffer_pose = current_pose
            # Get camera frames
            color_image, depth_image = self.get_frames()
            if color_image is None or depth_image is None:
                self.get_logger().error("Failed to get camera frames")
                return False
            
            # Show camera frames
            cv2.imshow("RealSense Color Image", color_image)
            #cv2.imshow("RealSense Depth Image", cv2.applyColorMap(
            #    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET))
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
                add_test_pose = input("Do you want to add a test pose? (y/n): ")
                if add_test_pose.lower() == 'y':
                    # Add a test pose
                    test_pose = Pose()
                    test_pose.position = Point(x=0.5, y=0.5, z=0.5)
                    test_pose.orientation = Quaternion(w=1, x=0, y=0, z=0)
                    fused_grasp = {'translation': [0.5, 0.5, 0.5]}
                else:
                    return False
            
            # Show results and wait for confirmation
            self.visualize_results(color_image, depth_image, langsam_result, 
                                 anygrasp_result, fused_grasp)
            
            
            # Calculate final robot pose
            target_pose = self.calculate_final_pose(fused_grasp, current_pose)
            self.get_logger().info(f"Calculated target pose: {target_pose}")

            # Change target pose z to be above the object take teh depth value from the anygrasp result
            target_pose.position.z = - fused_grasp['translation'][2]  + self.gripper_offset
            # target_pose.position.z =  -self.gripper_offset 
            #target_pose.position.x -= current_base_pose
            # Add base height to the target pose


            target_pose.position.x = target_pose.position.x #- 0.03
            target_pose.position.y = target_pose.position.y #- 0.03

            target_pose.position.x = round(target_pose.position.x, 7)
            target_pose.position.y = round(target_pose.position.y, 7)
            target_pose.position.z += current_pose.position.z
            #target_pose.position.z = target_pose.position.z   
            target_pose.position.z = round(target_pose.position.z, 7)

            self.get_logger().info(f"Current Base Pose z: {current_base_pose.position.z}")
            self.get_logger().info(f"Current End Effector Pose z: {current_pose.position.z}")
            self.get_logger().info(f"Calculated target pose z: {target_pose.position.z}")
            self.get_logger().info(f"Fused Grasp Translation z: {fused_grasp['translation'][2]}")


            # Update target poses with current psoes
            # target_pose.position.x = current_pose.position.x
            # target_pose.position.y = current_pose.position.y
            # target_pose.position.z = current_pose.position.z + 0.5
            self.get_logger().info(f"Calculated target pose Before Orientation Update: {target_pose}")
            # Change target pose orientation to be the same as the current pose - For testing (This needs to be corrected and checked)
            target_pose.orientation = current_pose.orientation 
            self.get_logger().info(f"Calculated Updated target pose: {target_pose}")

            self.create_pick_point_marker(target_pose)
            
            # Control robot
            await self.control_gripper("open")
            # Plan to pose
            if self.cartesian_path:
                trajectory = self.robot_control.plan_to_pose_cartesian(target_pose)
                # Additonal params for plan to pose cartesian step_size, jump_threshold
            else:
                trajectory = self.robot_control.plan_to_pose(target_pose)
            if trajectory:
                execute_input = input("Do you want to execute the Pick trajectory? (y/n): ")
                if execute_input.lower() == 'y':
                    if self.robot_control.execute_trajectory(trajectory):
                        await self.control_gripper("close")
                        self.get_logger().info("Pick operation completed")
                        #return True
            
            # self.get_logger().error("Failed to execute pick operation")        

            # Move Up
            self.get_client_names_and_types_by_node
            current_pose_1 = target_pose
            target_pose = buffer_pose

            if self.cartesian_path:
                trajectory = self.robot_control.plan_to_pose_cartesian(target_pose)
                # Additonal params for plan to pose cartesian step_size, jump_threshold
            else:
                trajectory = self.robot_control.plan_to_pose(target_pose)

            if trajectory:
                execute_input = input("Do you want to execute the Place trajectory? (y/n): ")
                if execute_input.lower() == 'y':
                    if self.robot_control.execute_trajectory(trajectory):
                        #await self.control_gripper("open")
                        self.get_logger().info("Move up operation completed")
                        #return True       
            #self.get_logger().error("Failed to execute move up operation")


            # Place Operation
            langsam_result_place = await self.process_place_api()
            if not langsam_result_place:
                self.get_logger().error("Failed to get place result")
                return False
            
            langsam_result_place = langsam_result_place['results'][0]
            target_pose_place = Pose()  
            target_pose_list = langsam_result_place['position_base']
            target_pose_place.position.x = target_pose_list[0]
            target_pose_place.position.y = target_pose_list[1]
            target_pose_place.position.z = current_pose_1.position.z + 0.1
            target_pose_place.orientation = current_pose_1.orientation


            self.get_logger().info(f"Calculated Place Pose: {target_pose_place}")

            if self.cartesian_path:
                trajectory = self.robot_control.plan_to_pose_cartesian(target_pose_place)
                # Additonal params for plan to pose cartesian step_size, jump_threshold
            else:
                trajectory = self.robot_control.plan_to_pose(target_pose_place)
            
            if trajectory:
                execute_input = input("Do you want to execute the Place trajectory? (y/n): ")
                if execute_input.lower() == 'y':
                    if self.robot_control.execute_trajectory(trajectory):
                        await self.control_gripper("open")
                        self.get_logger().info("Place operation completed")
                        # Open gripper
                        return True
            self.get_logger().error("Failed to execute place operation")
            return False


        except Exception as e:
            self.get_logger().error(f'Error in pick and place operation: {e}')
            self.get_logger().error(traceback.format_exc())
            return False
        finally:
            self.remove_marker()

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