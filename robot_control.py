import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.msg import DisplayTrajectory, MotionPlanRequest, WorkspaceParameters, Constraints, RobotState, RobotState, PositionConstraint, OrientationConstraint, JointConstraint
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetMotionPlan
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from moveit_msgs.msg import PositionConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
import threading
# Cartesian Path imports 
from moveit_msgs.srv import GetCartesianPath
#from moveit2.move_group_interface import MoveGroupInterface

class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control')
        self.callback_group = ReentrantCallbackGroup()
        
        # Service clients
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk', callback_group=self.callback_group)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik', callback_group=self.callback_group)
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path', callback_group=self.callback_group)
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path', callback_group=self.callback_group)
        
        # Publishers
        self.display_trajectory_pub = self.create_publisher(DisplayTrajectory, '/display_planned_path', 10)
        
        # Subscribers
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.current_joint_state = None

        # Action client for trajectory execution
        #self.trajectory_client = ActionClient(self, FollowJointTrajectory, '/follow_joint_trajectory')
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )
        
        # Check if the action server is available
        self.check_action_client_availability() 

        #self.move_group_interface = MoveGroupInterface("ur_manipulator", "base_link", self)

    def check_action_client_availability(self):
        self.get_logger().info('Checking scaled joint trajectory controller availability...')
        
        # Use a separate thread to check availability
        thread = threading.Thread(target=self._check_availability)
        thread.start()
        thread.join(timeout=6.0)  # Wait for up to 6 seconds
        
        if thread.is_alive():
            self.get_logger().error('Timed out while checking for scaled joint trajectory controller availability')
            return
        
        if hasattr(self, '_action_client_available'):
            if self._action_client_available:
                self.get_logger().info('Scaled joint trajectory controller is available.')
            else:
                self.get_logger().warn('Scaled joint trajectory controller is not available.')
                self.get_logger().warn('Please check the following:')
                self.get_logger().warn('1. Ensure that the robot driver is running')
                self.get_logger().warn('2. Verify that the scaled_joint_trajectory_controller is loaded and running')
                self.get_logger().warn('3. Check if the action server name is correct')
                self.get_logger().warn('4. Look for any error messages in the robot driver or controller logs')
        else:
            self.get_logger().error('Failed to determine action client availability')

    def _check_availability(self):
        self._action_client_available = self.trajectory_client.wait_for_server(timeout_sec=5.0)

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def get_current_base_pose(self):
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('FK service not available, waiting...')

        if self.current_joint_state is None:
            self.get_logger().error('No joint state received yet. Make sure the robot is publishing joint states.')
            return None

        request = GetPositionFK.Request()
        request.header = Header()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = 'base' # base_link
        request.fk_link_names = ['base_link']
        request.robot_state.joint_state = self.current_joint_state

        #self.get_logger().info(f"Requesting FK for joints: {request.robot_state.joint_state.name}")
        #self.get_logger().info(f"Joint positions: {request.robot_state.joint_state.position}")

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

    def get_current_ee_pose(self):
        while not self.fk_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('FK service not available, waiting...')

        if self.current_joint_state is None:
            self.get_logger().error('No joint state received yet. Make sure the robot is publishing joint states.')
            return None

        request = GetPositionFK.Request()
        request.header = Header()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = 'base_link'  # Adjust as needed #base_link
        request.fk_link_names = ['tool0']  # Adjust to your robot's end-effector link name #['wrist_3_link'] 
        request.robot_state.joint_state = self.current_joint_state

        #self.get_logger().info(f"Requesting FK for joints: {request.robot_state.joint_state.name}")
        #self.get_logger().info(f"Joint positions: {request.robot_state.joint_state.position}")
 
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

    def plan_to_pose(self, target_pose, planner_id="RRTConnectkConfigDefault", num_planning_attempts=5000, planning_time=30.0):
        while not self.plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().info('Motion planning service not available, waiting...')

        request = GetMotionPlan.Request()
        motion_request = MotionPlanRequest()

        # Set the start state
        motion_request.start_state.joint_state = self.current_joint_state

        # Set the goal constraints
        goal_constraints = Constraints()
        goal_constraints.name = "goal"

        # Create PoseStamped from target_pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose = target_pose

        # Use existing functions to create constraints
        position_constraint = self.create_position_constraint(pose_stamped)
        orientation_constraint = self.create_orientation_constraint(pose_stamped)

        # orientation_constraint1 = self.create_orientation_constraint1(pose_stamped)
        # orientation_constraint2 = self.create_orientation_constraint2(pose_stamped)

        goal_constraints.position_constraints.append(position_constraint)
        goal_constraints.orientation_constraints.append(orientation_constraint)

        # Add joint angle limits
        # joint_limits = [
        #     ("shoulder_pan_joint", -1.57, 1.57),  # -180 to 180 degrees
        #     ("shoulder_lift_joint", -1.57, 1.57),    # -180 to 0 degrees
        #     ("elbow_joint", 0, 1.57),             # 0 to 180 degrees
        #     ("wrist_1_joint", -1.57, 1.57),       # -180 to 180 degrees
        #     ("wrist_2_joint", -1.57, 1.57),       # -180 to 180 degrees
        #     ("wrist_3_joint", -1.57, 1.57)        # -180 to 180 degrees
        # ]
        # joint_limits = [
        #     ("elbow_joint", 0, 1.57),             # 0 to 180 degrees
        # ]

        # for joint_name, lower_limit, upper_limit in joint_limits:
        #     joint_constraint = JointConstraint()
        #     joint_constraint.joint_name = joint_name
        #     joint_constraint.position = (lower_limit + upper_limit) / 2  # Set to middle of range
        #     joint_constraint.tolerance_above = upper_limit - joint_constraint.position
        #     joint_constraint.tolerance_below = joint_constraint.position - lower_limit
        #     joint_constraint.weight = 1.0
        #     goal_constraints.joint_constraints.append(joint_constraint)

        motion_request.goal_constraints.append(goal_constraints)

        # Set the planner ID
        motion_request.planner_id = planner_id

        # Set the group name
        motion_request.group_name = "ur_manipulator"  # Adjust as needed for your robot

        # Set the number of planning attempts
        motion_request.num_planning_attempts = num_planning_attempts

        # Set the allowed planning time
        motion_request.allowed_planning_time = planning_time

        # Set the maximum velocity scaling factor
        motion_request.max_velocity_scaling_factor = 0.1  # Adjust as needed

        # Set the maximum acceleration scaling factor
        motion_request.max_acceleration_scaling_factor = 0.1 # Adjust as needed


        request.motion_plan_request = motion_request

        self.get_logger().info(f"Sending planning request to {motion_request.group_name} group using {planner_id} planner")
        self.get_logger().info(f"Target pose: {target_pose}")

        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.motion_plan_response.error_code.val == 1:  # Success
                trajectory = response.motion_plan_response.trajectory
                self.get_logger().info("Motion planning succeeded")
                self.display_trajectory(trajectory)
                return trajectory
            else:
                self.get_logger().error(f'Motion planning failed with error code: {response.motion_plan_response.error_code.val}')
                self.diagnose_planning_failure(response.motion_plan_response.error_code.val)
                return None
        else:
            self.get_logger().error('Failed to call motion planning service')
            return None

    def diagnose_planning_failure(self, error_code):
        if error_code == -1:
            self.get_logger().error("Planning failed: Unknown reason")
        elif error_code == -2:
            self.get_logger().error("Planning failed: Invalid motion plan request")
        elif error_code == -3:
            self.get_logger().error("Planning failed: Unable to initialize planner")
        elif error_code == -4:
            self.get_logger().error("Planning failed: Unable to extract path from planner")
        elif error_code == 9999:
            self.get_logger().error("Planning failed: No solution found. The target might be unreachable or in collision.")
            self.get_logger().info("Try the following:")
            self.get_logger().info("1. Check if the target pose is within the robot's workspace")
            self.get_logger().info("2. Ensure there are no collisions in the scene")
            self.get_logger().info("3. Increase the allowed planning time")
            self.get_logger().info("4. Try a different planner (e.g., 'RRTstar', 'PRMstar')")
        else:
            self.get_logger().error(f"Planning failed with unexpected error code: {error_code}")

    # def create_position_constraint(self, pose_stamped):
    #     from moveit_msgs.msg import PositionConstraint, BoundingVolume
    #     from shape_msgs.msg import SolidPrimitive

    #     constraint = PositionConstraint()
    #     constraint.header = pose_stamped.header
    #     constraint.link_name = 'wrist_3_link' #"tool0"  # Adjust as needed for your robot's end-effector link

    #     # Constraint region
    #     constraint.constraint_region.primitives.append(SolidPrimitive())
    #     constraint.constraint_region.primitives[0].type = SolidPrimitive.SPHERE
    #     constraint.constraint_region.primitives[0].dimensions = [0.005]  # 1cm sphere for position tolerance

    #     constraint.constraint_region.primitive_poses.append(pose_stamped.pose)

    #     constraint.weight = 1.0

    #     return constraint
    
    # Create Box constraint
    def create_position_constraint(self, pose_stamped):
        from moveit_msgs.msg import PositionConstraint, BoundingVolume
        from shape_msgs.msg import SolidPrimitive

        constraint = PositionConstraint()
        constraint.header = pose_stamped.header
        constraint.link_name = 'tool0'
        constraint.constraint_region.primitives.append(SolidPrimitive())
        constraint.constraint_region.primitives[0].type = SolidPrimitive.BOX
        constraint.constraint_region.primitives[0].dimensions = [0.1, 0.1, 0.1]
        constraint.constraint_region.primitive_poses.append(pose_stamped.pose)
        constraint.weight = 1.0

        return constraint

    def create_orientation_constraint(self, pose_stamped):
        from moveit_msgs.msg import OrientationConstraint

        constraint = OrientationConstraint()
        constraint.header = pose_stamped.header
        constraint.orientation = pose_stamped.pose.orientation
        constraint.link_name = 'tool0' #"tool0"  # Adjust as needed for your robot's end-effector link
        constraint.absolute_x_axis_tolerance = 0.1  # 1.14 degrees
        constraint.absolute_y_axis_tolerance = 0.1  # 1.14 degrees
        constraint.absolute_z_axis_tolerance = 0.1  # 1.14 degrees
        constraint.weight = 1.0

        return constraint
    
    def plan_to_pose_cartesian(self, target_pose, step_size=0.01, jump_threshold=0.0):
        """
        Plan a cartesian path to a target pose.
        
        Args:
            target_pose (Pose): Target pose for the end effector
            step_size (float): Maximum distance between waypoints (meters)
            jump_threshold (float): Maximum allowed joint value jump between consecutive points
            
        Returns:
            RobotTrajectory or None: Planned trajectory if successful, None otherwise
        """
        while not self.cartesian_path_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Cartesian path service not available, waiting...')
        
        if self.current_joint_state is None:
            self.get_logger().error('No joint state received yet')
            return None
            
        request = GetCartesianPath.Request()
        request.header = Header()
        request.header.frame_id = 'base_link'
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.joint_state = self.current_joint_state
        request.group_name = "ur_manipulator"
        request.link_name = "tool0"  # or wrist_3_link depending on your setup
        request.waypoints = [target_pose]
        request.max_step = step_size
        request.jump_threshold = jump_threshold
        request.avoid_collisions = True
        request.path_constraints.name = "path_constraints"
        
        self.get_logger().info(f"Planning cartesian path to target pose: {target_pose}")
        future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            if response.fraction > 0.95:  # Path found for at least 95% of the distance
                self.get_logger().info(f"Cartesian path planning succeeded, fraction: {response.fraction}")
                self.display_trajectory(response.solution)
                return response.solution
            else:
                self.get_logger().error(f'Could only compute {response.fraction * 100}% of the path')
                self.diagnose_cartesian_planning_failure(response.fraction)
                return None
        else:
            self.get_logger().error('Failed to call cartesian path planning service')
            return None

    def diagnose_cartesian_planning_failure(self, fraction):
        """Provide diagnostic information for cartesian planning failures"""
        self.get_logger().error(f"Cartesian planning failed, only achieved {fraction * 100}% of the path")
        self.get_logger().info("Common causes of cartesian planning failures:")
        self.get_logger().info("1. Target pose is outside the robot's reachable workspace")
        self.get_logger().info("2. Straight-line path would exceed joint limits")
        self.get_logger().info("3. Collision detected along the path")
        self.get_logger().info("4. IK solutions not found for some waypoints")
        self.get_logger().info("\nPossible solutions:")
        self.get_logger().info("1. Try reducing step_size for finer motion planning")
        self.get_logger().info("2. Adjust the jump_threshold if joint motions are too large")
        self.get_logger().info("3. Consider using regular plan_to_pose() instead")
        self.get_logger().info("4. Check if target pose is within workspace")

    def display_trajectory(self, trajectory):
        display_trajectory = DisplayTrajectory()
        display_trajectory.trajectory.append(trajectory)
        
        # Set the start state
        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        display_trajectory.trajectory_start = start_state
        
        # Set the correct model ID (robot name)
        display_trajectory.model_id = "ur"  # Change this to your robot's name as defined in the URDF

        self.get_logger().info(f"Publishing trajectory with {len(trajectory.joint_trajectory.points)} points")
        
        # Publish the trajectory multiple times to ensure it's received and displayed
        for _ in range(5):
            self.display_trajectory_pub.publish(display_trajectory)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Trajectory published for display in RViz")

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

        # 3. Plan to a new end effector position using MoveIt planner
        target_pose = Pose()
        # Increase height by 0.2 with respect to the current position
        target_pose.position.z = current_ee_pose.position.z + 0.4        # Reamining all the same
        target_pose.position.x = current_ee_pose.position.x
        target_pose.position.y = current_ee_pose.position.y
        target_pose.orientation = current_ee_pose.orientation

        robot_control.get_logger().info(f"Planning to target pose: {target_pose}")

        # 4. Plan and plot the trajectory in RViz
        trajectory = robot_control.plan_to_pose(target_pose)
        if trajectory:
            robot_control.get_logger().info("Trajectory planned successfully")
            robot_control.display_trajectory(trajectory)
            robot_control.get_logger().info("Trajectory should now be visible in RViz")
            
            # 5. Execute the trajectory

            execute_input = input("Do you want to execute the trajectory? (y/n): ")
            if execute_input.lower() == 'y':
                execute_success = robot_control.execute_trajectory(trajectory)
                if execute_success:
                    robot_control.get_logger().info("Trajectory execution completed successfully")
                else:
                    robot_control.get_logger().warn("Trajectory execution   failed")
            # execute_success = robot_control.execute_trajectory(trajectory)
            # if execute_success:
            #     robot_control.get_logger().info("Trajectory execution completed successfully")
            # else:
            #     robot_control.get_logger().warn("Trajectory execution failed")
        else:
            robot_control.get_logger().warn("Failed to plan trajectory")

        # Keep the node running to allow RViz to continue displaying the trajectory
        robot_control.get_logger().info("Keep this node running and check RViz for the displayed trajectory")
        rclpy.spin(robot_control)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        robot_control.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()