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

def calculate_orientation_to_point(current_pos, target_point):
    """
    Calculate quaternion orientation to make end effector face a target point
    
    Args:
        current_pos: Current position (Point)
        target_point: Point to look at (Point)
    Returns:
        Quaternion orientation
    """
    # Calculate direction vector from current position to target
    dx = target_point.x - current_pos.x
    dy = target_point.y - current_pos.y
    dz = target_point.z - current_pos.z
    
    # Calculate direction vector
    direction = np.array([dx, dy, dz])
    direction = direction / np.linalg.norm(direction)
    
    # Calculate rotation matrix
    z_axis = direction  # Camera looks along z-axis
    x_axis = np.array([1.0, 0.0, 0.0])  # Initial guess for x-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    
    # Convert rotation matrix to euler angles
    rotation_matrix = np.vstack((x_axis, y_axis, z_axis)).T
    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    pitch = math.atan2(-rotation_matrix[2, 0], 
                      math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    # Convert to quaternion
    q = euler2quat(roll, pitch, yaw)
    orientation = Quaternion()
    orientation.x = q[1]  # Note: euler2quat returns [w, x, y, z]
    orientation.y = q[2]
    orientation.z = q[3]
    orientation.w = q[0]
    
    return orientation

def generate_scanning_poses(start_pose, scan_radius, height_offset=0.3, num_points=20):
    """
    Generate poses for scanning around an object with end effector always facing the object
    Args:
        start_pose: Initial robot pose 
        scan_radius: Radius of the circular path
        height_offset: Height above the object for scanning
        num_points: Number of viewpoints
    """
    poses = []
    # Generate a circular path of points
    angles = np.linspace(0, 2*math.pi, num_points, endpoint=False)
    
    # Calculate object center (use start pose as reference)
    object_center = Point()
    object_center.x = start_pose.position.x
    object_center.y = start_pose.position.y
    object_center.z = start_pose.position.z - height_offset  # Object is below start position
    
    # Generate poses around the object
    for angle in angles:
        pose = Pose()
        
        # Calculate position on circular path
        pose.position.x = object_center.x + scan_radius * math.cos(angle)
        pose.position.y = object_center.y + scan_radius * math.sin(angle)
        pose.position.z = object_center.z + height_offset  # Maintain constant height above object
        
        # Calculate orientation to face object
        pose.orientation = calculate_orientation_to_point(pose.position, object_center)
        poses.append(pose)
    
    return poses, object_center

class ObjectScanningMotion(Node):
    def __init__(self, robot_control):
        super().__init__('object_scanning_motion')
        self.robot = robot_control
        self.logger = self.get_logger()
        
    def verify_pose_alignment(self, pose, object_center, threshold=0.01):
        """
        Verify that the pose's orientation points towards the object center
        within the given threshold
        
        Args:
            pose: Robot pose to verify
            object_center: Target point the end effector should face
            threshold: Maximum allowed deviation in meters
        Returns:
            bool: True if alignment is within threshold
        """
        # Get forward vector from orientation quaternion
        q = pose.orientation
        # Convert quaternion to rotation matrix and get forward vector (z-axis)
        forward_vector = np.array([
            2*(q.x*q.z + q.w*q.y),
            2*(q.y*q.z - q.w*q.x),
            1 - 2*(q.x*q.x + q.y*q.y)
        ])
        forward_vector = forward_vector / np.linalg.norm(forward_vector)
        
        # Calculate vector from pose to object center
        to_center = np.array([
            object_center.x - pose.position.x,
            object_center.y - pose.position.y,
            object_center.z - pose.position.z
        ])
        to_center = to_center / np.linalg.norm(to_center)
        
        # Calculate angle between vectors
        angle = np.arccos(np.clip(np.dot(forward_vector, to_center), -1.0, 1.0))
        
        # Convert angle to distance at the scanning radius
        distance = math.tan(angle) * np.linalg.norm([
            object_center.x - pose.position.x,
            object_center.y - pose.position.y,
            object_center.z - pose.position.z
        ])
        
        return distance <= threshold

    def execute_scanning_motion(self, scan_radius=0.3, height_offset=0.3, num_points=20):
        """
        Execute scanning motion around an object with end effector always facing the object
        
        Args:
            scan_radius (float): Radius of the circular path
            height_offset (float): Height above the object for scanning
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
                height_offset,
                num_points
            )

            self.logger.info(f"Starting scanning motion")
            self.logger.info(f"Object center: x={object_center.x:.3f}, y={object_center.y:.3f}, z={object_center.z:.3f}")
            self.logger.info(f"Scan radius: {scan_radius}m, Height offset: {height_offset}m")
            
            # Verify all poses point to object center
            for i, pose in enumerate(scanning_poses):
                if not self.verify_pose_alignment(pose, object_center):
                    self.logger.warning(f"Pose {i} alignment exceeds threshold")
            
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

    def wait_for_robot_init(self, timeout=10.0):
        """Wait for robot to initialize within timeout period"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.robot.current_joint_state is not None:
                self.logger.info("Robot initialized successfully")
                return True
            time.sleep(0.1)
        return False

def main(args=None):
    """Main function to run the scanning motion"""
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
        
        # Execute scanning motion
        success = scanner.execute_scanning_motion(
            scan_radius=0.3,    # 30cm radius for circular path
            height_offset=0.3,  # 30cm above object
            num_points=20       # 20 viewpoints
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