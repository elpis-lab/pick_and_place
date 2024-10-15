import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class TrajectorySender(Node):
    def __init__(self):
        super().__init__('ur10_trajectory_sender')
        self.action_client = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')
        
        # Wait until the action server is available
        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for action server...')

        self.send_trajectory()

    def send_trajectory(self):
        goal_msg = FollowJointTrajectory.Goal()
        
        # Create the JointTrajectory
        msg = JointTrajectory()
        msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        point = JointTrajectoryPoint()
        point.positions = [0.0, -1.0, 1.5, -1.0, 0.0, 0.0]  # Target positions in radians
        point.velocities = [0.05] * 6  # Target velocities
        point.time_from_start.sec = 2  # Duration to reach the target
        
        msg.points.append(point)
        
        # Assign the trajectory to the goal message
        goal_msg.trajectory = msg

        # Send the goal to the action server
        self.action_client.send_goal_async(goal_msg)
        self.get_logger().info('Sent trajectory command.')

def main(args=None):
    rclpy.init(args=args)
    trajectory_sender = TrajectorySender()
    rclpy.spin(trajectory_sender)
    trajectory_sender.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
