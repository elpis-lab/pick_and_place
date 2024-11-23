import gradio as gr
import cv2
import numpy as np
import asyncio
import rclpy
from threading import Thread
from rclpy.executors import MultiThreadedExecutor
from integrated_pick_place import IntegratedPickPlace
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import DisplayTrajectory
from functools import partial
import time
import base64
from rclpy.node import Node

class PickPlaceGUI():
    def __init__(self):
        # Initialize ROS
        rclpy.init()
        self.pick_place_system = IntegratedPickPlace()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.pick_place_system.robot_control)
        self.executor.add_node(self.pick_place_system)
        
        # Start executor in separate thread
        self.executor_thread = Thread(target=self.executor.spin)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        # Store current state
        self.current_color_image = None
        self.current_depth_image = None
        self.current_visualization = None
        self.current_pick_trajectory = None
        self.current_retry_pick_trajectory = None
        self.current_place_trajectory = None
        self.current_moveup_trajectory = None
        self.current_home_trajectory = None
        self.current_target_pose = None
        self.buffer_pose = None
        self.current_base_pose = None
        self.fused_grasp = None
        self.current_pose_1 = None
        self.gripper_state = "closed"
        self.running = True
        
        # Store place API visualization
        self.place_visualization = None

        # Create display trajectory publisher
        self.display_trajectory_publisher = self.pick_place_system.create_publisher(
            DisplayTrajectory,
            'display_planned_path',
            10
        )

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if hasattr(self, 'executor'):
            self.executor.shutdown()
        rclpy.shutdown()

    def async_handler(self, async_func, *args):
        """Wrapper to handle async functions in Gradio"""
        async def wrapper():
            loop = asyncio.get_event_loop()
            return await loop.create_task(async_func(*args))
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(wrapper())


    async def initialize_robot(self):
        """Initialize robot and get initial poses"""
        try:
            for _ in range(10):
                await asyncio.sleep(0.1)
                if self.pick_place_system.robot_control.current_joint_state is not None:
                    break

            current_pose = self.pick_place_system.robot_control.get_current_ee_pose()
            current_base_pose = self.pick_place_system.robot_control.get_current_base_pose()
            
            if not current_pose:
                return "Failed to get current robot pose"
                
            self.buffer_pose = current_pose
            self.current_base_pose = current_base_pose
            return "Robot initialized successfully"
        except Exception as e:
            return f"Error initializing robot: {str(e)}"

    async def emergency_gripper_control(self, action):
        """Emergency gripper control"""
        try:
            await self.pick_place_system.control_gripper(action)
            self.gripper_state = action
            return f"Emergency gripper {action} completed"
        except Exception as e:
            return f"Emergency gripper control failed: {str(e)}"

    def capture_frames(self):
        """Capture frames from RealSense camera"""
        try:
            color_image, depth_image = self.pick_place_system.get_frames()
            if color_image is not None and depth_image is not None:
                self.current_color_image = color_image.copy()
                self.current_depth_image = depth_image.copy()
                
                # Convert depth image to colormap for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )
                
                # Convert BGR to RGB for Gradio display
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                depth_colormap_rgb = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
                
                return color_image_rgb, depth_colormap_rgb, "Frames captured successfully"
            return None, None, "Failed to capture frames"
        except Exception as e:
            return None, None, f"Error capturing frames: {str(e)}"

    async def process_vision(self):
        """Process vision pipeline and return visualization"""
        try:
            if self.current_color_image is None or self.current_depth_image is None:
                return None, "No frames available. Please capture frames first."
                
            # Process with vision APIs
            langsam_result = await self.pick_place_system.process_langsam(self.current_color_image)
            anygrasp_result = await self.pick_place_system.process_anygrasp(
                self.current_color_image, 
                self.current_depth_image
            )
            
            # Fuse results
            self.fused_grasp = self.pick_place_system.fuse_grasps(langsam_result, anygrasp_result)
            
            # Create visualization
            viz = self.pick_place_system.visualize_results(
                self.current_color_image.copy(),
                self.current_depth_image.copy(),
                langsam_result,
                anygrasp_result,
                self.fused_grasp
            )
            
            # Convert visualization to RGB for Gradio display
            if viz is not None:
                viz_rgb = cv2.cvtColor(viz, cv2.COLOR_BGR2RGB)
            else:
                viz_rgb = None
            
            if not self.fused_grasp:
                return viz_rgb, "Failed to find suitable grasp"
                
            return viz_rgb, "Vision processing complete"
        except Exception as e:
            return None, f"Error in vision processing: {str(e)}"

    def add_test_pose(self):
        """Add a test pose when grasp detection fails"""
        try:
            self.fused_grasp = {'translation': [0.5, 0.5, 0.5]}
            test_pose = Pose()
            test_pose.position = Point(x=0.5, y=0.5, z=0.5)
            test_pose.orientation = Quaternion(w=1, x=0, y=0, z=0)
            self.current_target_pose = test_pose
            return "Test pose added successfully"
        except Exception as e:
            return f"Error adding test pose: {str(e)}"

    def calculate_pick_pose(self):
        """Calculate pick pose based on grasp"""
        try:
            if not self.fused_grasp:
                return "No grasp point available"

            current_pose = self.pick_place_system.robot_control.get_current_ee_pose()
            target_pose = self.pick_place_system.calculate_final_pose(self.fused_grasp, current_pose)

            target_pose.position.z = -self.fused_grasp['translation'][2] + self.pick_place_system.gripper_offset
            target_pose.position.x = round(target_pose.position.x, 7) 
            target_pose.position.y = round(target_pose.position.y, 7) 
            target_pose.position.z += current_pose.position.z
            target_pose.position.z = round(target_pose.position.z, 7)
            target_pose.orientation = current_pose.orientation
            
            self.current_target_pose = target_pose
            self.pick_place_system.create_pick_point_marker(target_pose)
            return f"Pick pose calculated: {target_pose}"
        except Exception as e:
            return f"Error calculating pick pose: {str(e)}"

    def display_trajectory_rviz(self, trajectory, trajectory_type=""):
        """Display trajectory in RViz"""
        try:
            if trajectory is None:
                return f"No {trajectory_type} trajectory available to display"
                
            display_trajectory = DisplayTrajectory()
            display_trajectory.trajectory.append(trajectory)
            display_trajectory.trajectory_start = self.pick_place_system.robot_control.get_current_state()
            
            self.display_trajectory_publisher.publish(display_trajectory)
            return f"{trajectory_type.capitalize()} trajectory displayed in RViz"
        except Exception as e:
            return f"Error displaying trajectory: {str(e)}"

    def plan_pick_trajectory(self, use_cartesian):
        """Plan pick trajectory"""
        try:
            if not self.current_target_pose:
                return "No target pose available", None
                
            if use_cartesian:
                trajectory = self.pick_place_system.robot_control.plan_to_pose_cartesian(
                    self.current_target_pose
                )
            else:
                trajectory = self.pick_place_system.robot_control.plan_to_pose(
                    self.current_target_pose
                )
                
            if trajectory:
                self.current_pick_trajectory = trajectory
                self.display_trajectory_rviz(trajectory, "pick")
                return "Pick trajectory planned successfully", trajectory
            return "Failed to plan pick trajectory", None
        except Exception as e:
            return f"Error planning pick trajectory: {str(e)}", None

    async def plan_retry_pick_trajectory(self, use_cartesian):
        """Plan retry pick trajectory"""
        try:
            new_pick_result = await self.pick_place_system.process_new_pick_api()
            if not new_pick_result:
                return "Failed to get new pick pose", None
                
            new_pick_result = new_pick_result['results'][0]
            target_pose = Pose()
            target_pose_list = new_pick_result['position_base']
            target_pose.position.x = target_pose_list[0]
            target_pose.position.y = target_pose_list[1]
            target_pose.position.z = (-self.fused_grasp['translation'][2] + 
                                    self.pick_place_system.gripper_offset + 
                                    self.buffer_pose.position.z)
            target_pose.orientation = self.buffer_pose.orientation
            
            self.current_target_pose = target_pose
            self.pick_place_system.update_marker_pose(target_pose)
            
            if use_cartesian:
                trajectory = self.pick_place_system.robot_control.plan_to_pose_cartesian(target_pose)
            else:
                trajectory = self.pick_place_system.robot_control.plan_to_pose(target_pose)
                
            if trajectory:
                self.current_retry_pick_trajectory = trajectory
                self.display_trajectory_rviz(trajectory, "retry pick")
                return f"Retry pick trajectory planned for pose: {target_pose}", trajectory
            return "Failed to plan retry pick trajectory", None
        except Exception as e:
            return f"Error planning retry pick: {str(e)}", None

    async def execute_pick_trajectory(self, is_retry=False):
        """Execute planned pick trajectory"""
        try:
            trajectory = self.current_retry_pick_trajectory if is_retry else self.current_pick_trajectory
            
            if not trajectory:
                return "No pick trajectory planned"
                
            await self.pick_place_system.control_gripper("open")
            time.sleep(3)
            if self.pick_place_system.robot_control.execute_trajectory(trajectory):
                await self.pick_place_system.control_gripper("close")
                self.current_pose_1 = self.current_target_pose
                return "Pick execution completed"
            return "Failed to execute pick trajectory"
        except Exception as e:
            return f"Error executing pick trajectory: {str(e)}"

    def plan_moveup_trajectory(self, use_cartesian):
        """Plan move up trajectory"""
        try:
            if use_cartesian:
                trajectory = self.pick_place_system.robot_control.plan_to_pose_cartesian(
                    self.buffer_pose
                )
            else:
                trajectory = self.pick_place_system.robot_control.plan_to_pose(
                    self.buffer_pose
                )
                
            if trajectory:
                self.current_moveup_trajectory = trajectory
                self.display_trajectory_rviz(trajectory, "moveup")
                return "Move up trajectory planned successfully", trajectory
            return "Failed to plan move up trajectory", None
        except Exception as e:
            return f"Error planning move up trajectory: {str(e)}", None

    async def execute_moveup_trajectory(self):
        """Execute planned move up trajectory"""
        try:
            if not self.current_moveup_trajectory:
                return "No move up trajectory planned"
                
            if self.pick_place_system.robot_control.execute_trajectory(self.current_moveup_trajectory):
                return "Move up execution completed"
            return "Failed to execute move up trajectory"
        except Exception as e:
            return f"Error executing move up trajectory: {str(e)}"

    async def plan_execute_place(self, use_cartesian):
        """Plan place trajectory"""
        try:
            place_result = await self.pick_place_system.process_place_api()
            if not place_result:
                return "Failed to get place location", None, None
                
            # Extract visualization from place API response
            place_viz_base64 = place_result.get('visualization_base64')
            place_viz = None
            if place_viz_base64:
                # Decode base64 to numpy array
                place_viz_bytes = base64.b64decode(place_viz_base64)
                place_viz_np = np.frombuffer(place_viz_bytes, dtype=np.uint8)
                place_viz = cv2.imdecode(place_viz_np, cv2.IMREAD_COLOR)
                # Convert BGR to RGB for Gradio display
                place_viz = cv2.cvtColor(place_viz, cv2.COLOR_BGR2RGB)
                self.place_visualization = place_viz
                
            place_data = place_result['results'][0]
            target_pose_place = Pose()
            target_pose_list = place_data['position_base']
            print("Place data: ", target_pose_list)
            target_pose_place.position.x = target_pose_list[0]
            target_pose_place.position.y = target_pose_list[1]
            target_pose_place.position.z = self.current_pose_1.position.z + 0.07
            target_pose_place.orientation = self.current_pose_1.orientation
            print("Target Pose Place: ", target_pose_place)
            if use_cartesian:
                trajectory = self.pick_place_system.robot_control.plan_to_pose_cartesian(
                    target_pose_place
                )
            else:
                trajectory = self.pick_place_system.robot_control.plan_to_pose(target_pose_place)
        
            if trajectory:
                self.current_place_trajectory = trajectory
                self.display_trajectory_rviz(trajectory, "place")
                print("Place trajectory planned successfully")
                return "Place trajectory planned successfully", trajectory, place_viz
            #self.logger.info(f"Place trajectory planning failed")
            print("Failed to plan place trajectory")
            return "Failed to plan place trajectory", None, place_viz
        except Exception as e:
            print(f"Error planning place trajectory: {str(e)}")
            return f"Error planning place trajectory: {str(e)}", None, None


    async def execute_place_trajectory(self):
        """Execute planned place trajectory"""
        try:
            if not self.current_place_trajectory:
                return "No place trajectory planned"
                
            if self.pick_place_system.robot_control.execute_trajectory(self.current_place_trajectory):
                await self.pick_place_system.control_gripper("open")
                return "Place execution completed"
            return "Failed to execute place trajectory"
        except Exception as e:
            return f"Error executing place trajectory: {str(e)}"

def create_gui():
    pick_place_gui = PickPlaceGUI()
    
    with gr.Blocks(title="Pick and Place Control Interface") as interface:
        gr.Markdown("# 🤖 Robot Pick and Place Control System - ELPIS")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input prompts
                prompt_input = gr.Textbox(
                    label="Pick Object Prompt",
                    value="screw_driver",
                    interactive=True
                )
                place_prompt_input = gr.Textbox(
                    label="Place Location Prompt",
                    value="plate",
                    interactive=True
                )
                
                # Emergency Gripper Control
                with gr.Row():
                    emergency_open_btn = gr.Button(
                        "🚨 Emergency Open Gripper",
                        variant="secondary"
                    )
                    emergency_close_btn = gr.Button(
                        "🚨 Emergency Close Gripper",
                        variant="secondary"
                    )
                
                # Initialize Robot
                init_btn = gr.Button("1. Initialize Robot", variant="primary")
                init_status = gr.Textbox(
                    label="Initialization Status",
                    interactive=False
                )
                
                # Vision Controls
                with gr.Row():
                    capture_btn = gr.Button("2. Capture Frames", variant="primary")
                    process_btn = gr.Button("3. Process Vision", variant="primary")
                test_pose_btn = gr.Button(
                    "3a. Add Test Pose (if vision fails)",
                    variant="secondary"
                )
                
                # Pick Controls
                calc_pick_btn = gr.Button("4. Calculate Pick Pose", variant="primary")
                pick_pose_status = gr.Textbox(
                    label="Pick Pose Status",
                    interactive=False
                )
                
                # Pick Trajectory Control
                with gr.Accordion("Pick Trajectory Control", open=False):
                    with gr.Row():
                        plan_pick_cartesian_btn = gr.Button(
                            "5a. Plan Pick (Cartesian)",
                            variant="primary"
                        )
                        plan_pick_rrt_btn = gr.Button(
                            "5b. Plan Pick (RRT)",
                            variant="primary"
                        )
                    show_pick_traj_btn = gr.Button(
                        "Show Pick Trajectory in RViz",
                        variant="secondary"
                    )
                    execute_pick_btn = gr.Button(
                        "Execute Pick Trajectory",
                        variant="primary"
                    )
                    pick_traj_status = gr.Textbox(
                        label="Pick Trajectory Status",
                        interactive=False
                    )
                
                # Retry Pick Control
                with gr.Accordion("Retry Pick Control", open=False):
                    with gr.Row():
                        plan_retry_cartesian_btn = gr.Button(
                            "6a. Plan Retry Pick (Cartesian)",
                            variant="primary"
                        )
                        plan_retry_rrt_btn = gr.Button(
                            "6b. Plan Retry Pick (RRT)",
                            variant="primary"
                        )
                    show_retry_traj_btn = gr.Button(
                        "Show Retry Trajectory in RViz",
                        variant="secondary"
                    )
                    execute_retry_btn = gr.Button(
                        "Execute Retry Pick Trajectory",
                        variant="primary"
                    )
                    retry_traj_status = gr.Textbox(
                        label="Retry Trajectory Status",
                        interactive=False
                    )

                # Move Up Control
                with gr.Accordion("Move Up Control", open=False):
                    with gr.Row():
                        plan_moveup_cartesian_btn = gr.Button(
                            "7a. Plan Move Up (Cartesian)",
                            variant="primary"
                        )
                        plan_moveup_rrt_btn = gr.Button(
                            "7b. Plan Move Up (RRT)",
                            variant="primary"
                        )
                    show_moveup_traj_btn = gr.Button(
                        "Show Move Up Trajectory in RViz",
                        variant="secondary"
                    )
                    execute_moveup_btn = gr.Button(
                        "Execute Move Up Trajectory",
                        variant="primary"
                    )
                    moveup_traj_status = gr.Textbox(
                        label="Move Up Trajectory Status",
                        interactive=False
                    )

                # Place Control
                with gr.Accordion("Place Control", open=False):
                    with gr.Row():
                        plan_place_cartesian_btn = gr.Button(
                            "8a. Plan Place (Cartesian)",
                            variant="primary"
                        )
                        plan_place_rrt_btn = gr.Button(
                            "8b. Plan Place (RRT)",
                            variant="primary"
                        )
                    show_place_traj_btn = gr.Button(
                        "Show Place Trajectory in RViz",
                        variant="secondary"
                    )
                    execute_place_btn = gr.Button(
                        "Execute Place Trajectory",
                        variant="primary"
                    )
                    place_traj_status = gr.Textbox(
                        label="Place Trajectory Status",
                        interactive=False
                    )
            
            # Right column for visual outputs
            with gr.Column(scale=1):
                # Camera feeds
                with gr.Row():
                    color_output = gr.Image(
                        label="Color Feed",
                        type="numpy"
                    )
                    depth_output = gr.Image(
                        label="Depth Feed",
                        type="numpy"
                    )
                
                # Vision processing output
                vision_output = gr.Image(
                    label="Vision Processing",
                    type="numpy"
                )

                # Place API visualization output
                place_viz_output = gr.Image(
                    label="Place Location Analysis",
                    type="numpy"
                )

                # Global status output
                status_output = gr.Textbox(
                    label="Operation Status",
                    interactive=False,
                    lines=3
                )

        # Event Handlers
        # Emergency Controls
        emergency_open_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.emergency_gripper_control,
                "open"
            ),
            outputs=[status_output]
        )
        
        emergency_close_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.emergency_gripper_control,
                "close"
            ),
            outputs=[status_output]
        )

        # Initialization
        init_btn.click(
            lambda: pick_place_gui.async_handler(pick_place_gui.initialize_robot),
            outputs=[init_status]
        )
        
        # Vision Processing
        capture_btn.click(
            pick_place_gui.capture_frames,
            outputs=[color_output, depth_output, status_output]
        )
        
        process_btn.click(
            lambda: pick_place_gui.async_handler(pick_place_gui.process_vision),
            outputs=[vision_output, status_output]
        )
        
        test_pose_btn.click(
            pick_place_gui.add_test_pose,
            outputs=[status_output]
        )
        
        # Pick Controls
        calc_pick_btn.click(
            pick_place_gui.calculate_pick_pose,
            outputs=[pick_pose_status]
        )
        
        # Pick Trajectory
        plan_pick_cartesian_btn.click(
            lambda: pick_place_gui.plan_pick_trajectory(True),
            outputs=[status_output, pick_traj_status]
        )
        
        plan_pick_rrt_btn.click(
            lambda: pick_place_gui.plan_pick_trajectory(False),
            outputs=[status_output, pick_traj_status]
        )
        
        show_pick_traj_btn.click(
            lambda: pick_place_gui.display_trajectory_rviz(
                pick_place_gui.current_pick_trajectory,
                "pick"
            ),
            outputs=[status_output]
        )
        
        execute_pick_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.execute_pick_trajectory,
                False
            ),
            outputs=[status_output]
        )
        
        # Retry Pick
        plan_retry_cartesian_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.plan_retry_pick_trajectory,
                True
            ),
            outputs=[status_output, retry_traj_status]
        )
        
        plan_retry_rrt_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.plan_retry_pick_trajectory,
                False
            ),
            outputs=[status_output, retry_traj_status]
        )
        
        show_retry_traj_btn.click(
            lambda: pick_place_gui.display_trajectory_rviz(
                pick_place_gui.current_retry_pick_trajectory,
                "retry pick"
            ),
            outputs=[status_output]
        )
        
        execute_retry_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.execute_pick_trajectory,
                True
            ),
            outputs=[status_output]
        )
        
        # Move Up
        plan_moveup_cartesian_btn.click(
            lambda: pick_place_gui.plan_moveup_trajectory(True),
            outputs=[status_output, moveup_traj_status]
        )
        
        plan_moveup_rrt_btn.click(
            lambda: pick_place_gui.plan_moveup_trajectory(False),
            outputs=[status_output, moveup_traj_status]
        )
        
        show_moveup_traj_btn.click(
            lambda: pick_place_gui.display_trajectory_rviz(
                pick_place_gui.current_moveup_trajectory,
                "moveup"
            ),
            outputs=[status_output]
        )
        
        execute_moveup_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.execute_moveup_trajectory
            ),
            outputs=[status_output]
        )
        
        # Place Trajectory - Updated with visualization
        plan_place_cartesian_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.plan_execute_place,
                True
            ),
            outputs=[status_output, place_traj_status, place_viz_output]
        )
        
        plan_place_rrt_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.plan_execute_place,
                False
            ),
            outputs=[status_output, place_traj_status, place_viz_output]
        )
        
        show_place_traj_btn.click(
            lambda: pick_place_gui.display_trajectory_rviz(
                pick_place_gui.current_place_trajectory,
                "place"
            ),
            outputs=[status_output]
        )
        
        execute_place_btn.click(
            lambda: pick_place_gui.async_handler(
                pick_place_gui.execute_place_trajectory
            ),
            outputs=[status_output]
        )

        # Input Changes
        prompt_input.change(
            lambda x: setattr(pick_place_gui.pick_place_system, 'prompt', x),
            inputs=[prompt_input]
        )
        
        place_prompt_input.change(
            lambda x: setattr(pick_place_gui.pick_place_system, 'place_prompt', x),
            inputs=[place_prompt_input]
        )

    return interface, pick_place_gui

if __name__ == "__main__":
    interface, pick_place_gui = create_gui()
    try:
        interface.launch(allowed_paths=["images/*"])
    finally:
        pick_place_gui.cleanup()