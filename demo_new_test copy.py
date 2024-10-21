import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control')
        self.callback_group = ReentrantCallbackGroup()
        
        # Service clients
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=self.callback_group)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=self.callback_group)
        
        # Publishers
        self.display_trajectory_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joint_state = None

        # Action client for trajectory execution
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/follow_joint_trajectory')

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def get_current_ee_pose(self):
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('FK service not available, waiting...')

        if self.current_joint_state is None:
            self.get_logger().error('No joint state received yet. Make sure the robot is publishing joint states.')
            return None

        request = GetPositionFK.Request()
        request.header = Header()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = 'base_link'  # Adjust as needed
        request.fk_link_names = ['tool0']  # Adjust to your robot's end-effector link name
        request.robot_state.joint_state = self.current_joint_state

        self.get_logger().info(f"Requesting FK for joints: {request.robot_state.joint_state.name}")
        self.get_logger().info(f"Joint positions: {request.robot_state.joint_state.position}")

        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            self.get_logger().info(f"FK service response: {result}")
            if result.pose_stamped:
                return result.pose_stamped[0].pose
            else:
                self.get_logger().error('FK service returned an empty result')
                return None
        else:
            self.get_logger().error('Failed to call FK service')
            return None

    def get_current_joint_angles(self):
        if self.current_joint_state:
            return dict(zip(self.current_joint_state.name, self.current_joint_state.position))
        else:
            self.get_logger().warn('No joint state received yet')
            return None

    def plan_to_pose(self, target_pose):
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('IK service not available, waiting...')

        request = GetPositionIK.Request()
        request.ik_request.group_name = 'ur_manipulator'  # Adjust as needed for your robot
        request.ik_request.pose_stamped.header.frame_id = 'base_link'  # Adjust as needed
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True

        # Set the start state to the current state
        request.ik_request.robot_state.joint_state = self.current_joint_state

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            solution = future.result().solution
            trajectory = self.create_trajectory(solution)
            self.display_trajectory(trajectory)
            return trajectory
        else:
            self.get_logger().error('Failed to call IK service')
            return None

    def create_trajectory(self, solution):
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = solution.joint_state.name
        
        # Add the start point (current state)
        start_point = JointTrajectoryPoint()
        start_point.positions = self.current_joint_state.position
        start_point.time_from_start.sec = 0
        trajectory.joint_trajectory.points.append(start_point)
        
        # Add the end point (goal state)
        end_point = JointTrajectoryPoint()
        end_point.positions = solution.joint_state.position
        end_point.time_from_start.sec = 1  # Adjust as needed
        trajectory.joint_trajectory.points.append(end_point)
        
        return trajectory

    def display_trajectory(self, trajectory):
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory.append(trajectory)
        
        # Set the start state
        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        display_trajectory.trajectory_start = start_state
        
        # Set the correct frame ID
        display_trajectory.model_id = "ur"  # Adjust if your robot model has a different name
        
        self.get_logger().info(f"Publishing trajectory with {len(trajectory.joint_trajectory.points)} points")
        self.display_trajectory_pub.publish(display_trajectory)

    def execute_trajectory(self, trajectory):
        # Wait for the action server to be available
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        # Create a FollowJointTrajectory action goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory.joint_trajectory

        # Send the goal
        self.get_logger().info('Sending trajectory execution goal')
        send_goal_future = self.trajectory_client.send_goal_async(goal_msg)

        # Wait for the server to accept the goal
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted')

        # Wait for the result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Trajectory execution succeeded')
            return True
        else:
            self.get_logger().error(f'Trajectory execution failed with error code: {result.error_code}')
            return False

def main(args=None):
    rclpy.init(args=args)
    robot_control = RobotControl()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_control)

    # Give more time for the node to initialize and receive joint states
    for _ in range(10):  # Try up to 10 times
        rclpy.spin_once(robot_control, timeout_sec=1.0)
        if robot_control.current_joint_state is not None:
            break
    else:
        robot_control.get_logger().error("Failed to receive joint states after 10 seconds")
        executor.shutdown()
        robot_control.destroy_node()
        rclpy.shutdown()
        return

    try:
        # 1. Get current robot position (Base to end effector)
        current_ee_pose = robot_control.get_current_ee_pose()
        if current_ee_pose:
            robot_control.get_logger().info(f"Current end-effector pose: {current_ee_pose}")
        else:
            robot_control.get_logger().warn("Failed to get current end-effector pose")

        # 2. Get current robot position (all joint angles)
        current_joint_angles = robot_control.get_current_joint_angles()
        if current_joint_angles:
            robot_control.get_logger().info(f"Current joint angles: {current_joint_angles}")
        else:
            robot_control.get_logger().warn("Failed to get current joint angles")

        # 3. Plan to a new end effector position
        target_pose = Pose()
        # target_pose.position.x = 0.5278
        # target_pose.position.y = 0.1100
        # target_pose.position.z = 0.4928
        # target_pose.orientation.x = 0.73091
        # target_pose.orientation.y = 0.6811
        # target_pose.orientation.z = 0.0231
        # target_pose.orientation.w = 0.036252
        # Add vertical z - 0.2 to current position
        target_pose.position.x = current_ee_pose.position.x
        target_pose.position.y = current_ee_pose.position.y
        target_pose.position.z = current_ee_pose.position.z + 0.2
        target_pose.orientation = current_ee_pose.orientation
        robot_control.get_logger().info(f"Planning to target pose: {target_pose}")

        # 4. Plot the trajectory in RViz
        trajectory = robot_control.plan_to_pose(target_pose)
        if trajectory:
            robot_control.get_logger().info("Trajectory planned and displayed in RViz")
            
            # 5. Execute the trajectory
            execute_success = robot_control.execute_trajectory(trajectory)
            if execute_success:
                robot_control.get_logger().info("Trajectory execution completed successfully")
            else:
                robot_control.get_logger().warn("Trajectory execution failed")
        else:
            robot_control.get_logger().warn("Failed to plan trajectory")

        # Keep the node running to allow RViz to display the trajectory
        rclpy.spin(robot_control)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        robot_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()