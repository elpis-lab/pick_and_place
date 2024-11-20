#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import math
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import threading

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # Publisher for actual trajectory execution
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )
        
        # Publisher for RViz visualization
        self.display_trajectory_pub = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10
        )
        
        # Subscriber for current joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.current_joint_state = None

        # Gripper API configuration
        self.gripper_api_url = "http://localhost:8005"
        
    def control_gripper(self, action):
        """Control the gripper through its API"""
        try:
            response = requests.post(f"{self.gripper_api_url}/{action}")
            return response.json()
        except Exception as e:
            self.get_logger().error(f'Error controlling gripper: {e}')
            return None

    def execute_gripper_action(self, action, delay=0.0):
        """Execute gripper action after specified delay"""
        def delayed_action():
            time.sleep(delay)
            self.control_gripper(action)
        
        # Start gripper action in a separate thread
        thread = threading.Thread(target=delayed_action)
        thread.start()

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def degrees_to_radians(self, degrees_list):
        return [math.radians(deg) for deg in degrees_list]
    
    def create_trajectory(self):
        trajectory = JointTrajectory()
        trajectory.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        # target_positions_deg = [-1.95,-90.52,136.92,-46.79,-90.51,-2.20]#start
        target_positions_deg = [-2.24, -81.98, 103.33, -144.47, -87.79, -6.28]#end
        target_positions_rad = self.degrees_to_radians(target_positions_deg)
        
        current_positions = list(self.current_joint_state.position)
        
        # duration_sec = 5.0
        # velocities = []
        # for curr, target in zip(current_positions, target_positions_rad):
        #     velocity = (target - curr) / duration_sec
        #     velocity = max(min(velocity, 1.0), -1.0)
        #     velocities.append(velocity)

        point = JointTrajectoryPoint()
        point.positions = target_positions_rad
        
        duration = Duration()
        duration.sec = 0
        duration.nanosec = 600000000
        point.time_from_start = duration

        trajectory.points = [point]
        return trajectory

    def visualize_trajectory(self, trajectory):
        if not self.current_joint_state:
            self.get_logger().error('No joint states received yet. Cannot visualize.')
            return False

        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = "ur"

        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        display_trajectory.trajectory_start = start_state

        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = trajectory
        display_trajectory.trajectory.append(robot_trajectory)

        for _ in range(5):
            self.display_trajectory_pub.publish(display_trajectory)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Trajectory has been visualized in RViz')
        return True

    def execute_trajectory(self):
        if not self.current_joint_state:
            self.get_logger().error('No joint states available. Cannot execute trajectory.')
            return

        trajectory = self.create_trajectory()
        if not self.visualize_trajectory(trajectory):
            return

        self.get_logger().info("\nCurrent joint positions (degrees):")
        for name, pos in zip(self.current_joint_state.name, self.current_joint_state.position):
            self.get_logger().info(f"{name}: {math.degrees(pos):.2f}")

        self.get_logger().info("\nTrajectory has been visualized in RViz.")
        self.get_logger().info("Do you want to:")
        self.get_logger().info("1. Execute trajectory with gripper open at start")
        self.get_logger().info("2. Execute trajectory with gripper open during motion (specify delay)")
        self.get_logger().info("3. Cancel")
        
        response = input("Enter choice (1/2/3): ").strip()
        
        if response == '1':
            self.get_logger().info('Opening gripper and executing trajectory...')
            self.control_gripper("open")
            time.sleep(0.03)  # Wait for gripper to open
            self.trajectory_pub.publish(trajectory)
        elif response == '2':
            delay = float(input("Enter delay in seconds for gripper opening (e.g., 0.3): "))
            self.get_logger().info(f'Executing trajectory with gripper opening after {delay} seconds...')
            self.execute_gripper_action("open", delay)
            self.trajectory_pub.publish(trajectory)
        else:
            self.get_logger().info("Trajectory execution cancelled")
            return

        self.get_logger().info('Trajectory sent to controller')

def main():
    rclpy.init()
    node = TrajectoryPublisher()

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.5)
        if node.current_joint_state is not None:
            break
    else:
        node.get_logger().error("Failed to receive joint states")
        node.destroy_node()
        rclpy.shutdown()
        return

    try:
        node.execute_trajectory()
    except KeyboardInterrupt:
        node.get_logger().info("Operation cancelled by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()