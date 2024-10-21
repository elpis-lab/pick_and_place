#!/usr/bin/env python3
# In DEVELOPMENT

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from tf.transformations import quaternion_from_euler
import copy

class UR10ControlHelper:
    def __init__(self, robot_name="ur10", ee_link="tool0"):
        # Initialize the node
        rospy.init_node('ur10_control_helper', anonymous=True)

        # Initialize the robot commander
        self.robot = moveit_commander.RobotCommander()

        # Initialize the scene
        self.scene = moveit_commander.PlanningSceneInterface()

        # Initialize the move group
        self.move_group = moveit_commander.MoveGroupCommander(robot_name)

        # Set the end effector link
        self.move_group.set_end_effector_link(ee_link)

        # Publisher to publish trajectories
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

    def load_scene(self, scene_urdf_path):
        """Load a URDF file into the planning scene."""
        with open(scene_urdf_path, 'r') as file:
            urdf_content = file.read()

        rospy.set_param('/scene_urdf', urdf_content)
        rospy.sleep(1)  # Give some time for the parameter to be set

        # Load the URDF into the planning scene
        success = self.scene.load_urdf("scene_object", urdf_content, pose=geometry_msgs.msg.Pose())
        if success:
            rospy.loginfo("Successfully loaded scene URDF")
        else:
            rospy.logerr("Failed to load scene URDF")

    def move_to_pose(self, x, y, z, roll, pitch, yaw):
        """Move the end effector to a specific pose."""
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        # Convert Euler angles to quaternion
        q = quaternion_from_euler(roll, pitch, yaw)
        pose_goal.orientation.x = q[0]
        pose_goal.orientation.y = q[1]
        pose_goal.orientation.z = q[2]
        pose_goal.orientation.w = q[3]

        self.move_group.set_pose_target(pose_goal)

        # Plan and execute
        plan = self.move_group.go(wait=True)
        
        # Clear targets after planning
        self.move_group.clear_pose_targets()

        return plan

    def move_to_joint_state(self, joint_goal):
        """Move to a specific joint state."""
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    def get_current_pose(self):
        """Get the current pose of the end effector."""
        return self.move_group.get_current_pose().pose

    def get_current_joint_values(self):
        """Get the current joint values."""
        return self.move_group.get_current_joint_values()

    def plan_cartesian_path(self, waypoints):
        """Plan a Cartesian path through a list of waypoints."""
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints,   # waypoints to follow
            0.01,        # eef_step
            0.0)         # jump_threshold

        return plan, fraction

    def execute_plan(self, plan):
        """Execute a saved plan."""
        self.move_group.execute(plan, wait=True)

    def shutdown(self):
        """Shut down MoveIt cleanly."""
        moveit_commander.roscpp_shutdown()

def main():
    try:
        ur10_helper = UR10ControlHelper()

        # Load the scene URDF
        ur10_helper.load_scene("/path/to/your/scene.urdf")

        # Move to a specific pose
        ur10_helper.move_to_pose(0.5, 0.1, 0.4, 0, pi/2, 0)

        # Get current pose
        current_pose = ur10_helper.get_current_pose()
        rospy.loginfo(f"Current pose: {current_pose}")

        # Plan and execute a Cartesian path
        waypoints = []
        wpose = ur10_helper.get_current_pose()
        wpose.position.z -= 0.1  # First move up (z)
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.y += 0.1  # Second move forward/backward in (y)
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.x -= 0.1  # Third move sideways (x)
        waypoints.append(copy.deepcopy(wpose))

        cartesian_plan, fraction = ur10_helper.plan_cartesian_path(waypoints)
        ur10_helper.execute_plan(cartesian_plan)

    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return

if __name__ == "__main__":
    main()