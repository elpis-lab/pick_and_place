import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
import math
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import time
import threading
from transforms3d.euler import euler2quat
from robot_control_test import RobotControl

def calculate_orientation_to_point(current_pos, target_point, start_orientation):
    """
    Calculate quaternion orientation to make end effector face a target point while maintaining
    similar orientation to start pose
    """
    # Calculate direction vector from current position to target
    dx = target_point.x - current_pos.x
    dy = target_point.y - current_pos.y
    dz = target_point.z - current_pos.z
    
    # Calculate yaw for facing the object
    yaw = math.atan2(dy, dx)
    
    # Calculate pitch to point towards object
    dist_xy = math.sqrt(dx*dx + dy*dy)
    pitch = -math.atan2(dz, dist_xy)
    
    # Keep roll from starting orientation (this helps maintain overall orientation)
    roll = 0  # You might want to extract roll from start_orientation if needed
    
    # Convert to quaternion
    q = euler2quat(roll, pitch, yaw)
    orientation = Quaternion()
    orientation.x = q[1]
    orientation.y = q[2]
    orientation.z = q[3]
    orientation.w = q[0]
    
    return orientation

def generate_scanning_poses(start_pose, scan_radius, num_points=20):
    """
    Generate poses for scanning around an object directly below start pose
    Args:
        start_pose: Initial robot pose
        scan_radius: Radius of the semicircular path
        num_points: Number of viewpoints
    """
    poses = []
    # Generate a semicircle of points (from -90 to +90 degrees)
    angles = np.linspace(-math.pi/2, math.pi/2, num_points)
    
    # Calculate object center (directly below start pose at 0.4m distance)
    object_center = Point()
    object_center.x = start_pose.position.x
    object_center.y = start_pose.position.y
    object_center.z = start_pose.position.z
    
    for angle in angles:
        pose = Pose()
        # Calculate position maintaining the same height and distance to object
        pose.position.x = object_center.x + 0.4 * math.cos(angle)  # Keep 0.4m distance
        pose.position.z = object_center.z - scan_radius * math.cos(angle)
        pose.position.y = start_pose.position.y  # Maintain original height
        
        # Calculate orientation to face object while maintaining similar orientation to start
        pose.orientation = calculate_orientation_to_point(pose.position, object_center, start_pose.orientation)
        poses.append(pose)

    for angle in angles:
        pose = Pose()
        # Calculate position maintaining the same height and distance to object
        pose.position.x = object_center.x - 0.4 * math.cos(angle)  # Keep 0.4m distance
        pose.position.z = object_center.z - scan_radius * math.cos(angle)
        pose.position.y = start_pose.position.y  # Maintain original height
        
        # Calculate orientation to face object while maintaining similar orientation to start
        pose.orientation = calculate_orientation_to_point(pose.position, object_center, start_pose.orientation)
        poses.append(pose)
    
    return poses, object_center

# Rest of the code remains exactly the same as your original code
class ObjectScanningMotion(Node):
    def __init__(self, robot_control):
        super().__init__('object_scanning_motion')
        self.robot = robot_control
        self.logger = self.get_logger()

    def wait_for_robot_init(self, timeout=10.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.robot.current_joint_state is not None:
                self.logger.info("Robot initialized successfully")
                return True
            time.sleep(0.1)
        return False

    def execute_scanning_motion(self, scan_radius=0.2, num_points=20):
        """
        Execute scanning motion around an object directly below robot
        
        Args:
            scan_radius (float): Radius of the semicircular path
            num_points (int): Number of scanning positions
        """
        try:
            if not self.wait_for_robot_init():
                self.logger.error("Robot initialization timeout")
                return False

            start_pose = self.robot.get_current_ee_pose()
            if not start_pose:
                self.logger.error("Failed to get current pose")
                return False

            scanning_poses, object_center = generate_scanning_poses(
                start_pose, 
                scan_radius,
                num_points
            )

            self.logger.info(f"Starting scanning motion")
            self.logger.info(f"Object center (calculated): x={object_center.x:.3f}, y={object_center.y:.3f}, z={object_center.z:.3f}")
            self.logger.info(f"Will capture {num_points} images in semicircular path")
            self.logger.info(f"Maintaining 0.4m distance to object with radius {scan_radius}m")

            for i, pose in enumerate(scanning_poses):
                self.logger.info(f"\nMoving to scanning position {i+1}/{len(scanning_poses)}")
                self.logger.info(f"Target position: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")
                
                traj = self.robot.plan_to_pose_cartesian(pose, step_size=0.01)
                if not traj:
                    self.logger.error(f"Failed to plan movement to scanning position {i+1}")
                    return False

                execute_input = input(f"Move to scanning position {i+1}? (y/n): ")
                if execute_input.lower() != 'y':
                    self.logger.info("Scanning motion cancelled by user")
                    return False
                
                if not self.robot.execute_trajectory(traj):
                    self.logger.error(f"Failed to execute movement to scanning position {i+1}")
                    return False

                self.logger.info(f"At position {i+1} - Ready to capture depth image")
                input("Press Enter after capturing depth image...")

            self.logger.info("Scanning motion completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error during scanning motion: {str(e)}")
            return False

def main(args=None):
    rclpy.init(args=args)
    
    print("Initializing Robot Control...")
    robot_control = RobotControl()
    
    executor = MultiThreadedExecutor()
    executor.add_node(robot_control)
    
    scanner = ObjectScanningMotion(robot_control)
    executor.add_node(scanner)
    
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    try:
        print("Waiting for robot to initialize...")
        time.sleep(2)
        
        success = scanner.execute_scanning_motion(
            scan_radius=0.2,  # 20cm radius for semicircle
            num_points=20     # 20 viewpoints
        )
        
        if success:
            scanner.get_logger().info("Scanning completed successfully")
        else:
            scanner.get_logger().error("Scanning failed")
            
    except KeyboardInterrupt:
        scanner.get_logger().info("Scanning interrupted by user")
    finally:
        executor.shutdown()
        robot_control.destroy_node()
        scanner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()