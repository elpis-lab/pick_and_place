from __future__ import print_function
import pybullet
import pybullet_data
import gym
import numpy as np
import math
import os
import pybullet as p
import time
import pandas as pd
from pybullet_tools.utils import (
    WorldSaver,
    HideOutput,
    connect,
    enable_preview,
    set_camera,
    Pose,
    Point,
    Euler,
    stable_z,
    set_pose,
    add_data_path,
    enable_gravity,
    disable_gravity,
    wait_if_gui,
    disconnect,
    get_movable_joints,
    set_joint_positions,
    simulate_for_sim_duration,
)
from robots.ur10_primitives import (
    BodyPose,
    BodyConf,
    Command,
    get_grasp_gen,
    get_ik_fn,
    get_free_motion_gen,
    get_holding_motion_gen,
)

class UR10(gym.Env):
    observation_space = gym.spaces.Dict(dict(
        desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        observation=gym.spaces.Box(-np.inf, np.inf, shape=(25,), dtype='float32'),
    ))

    action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
    position_bounds = [(0.5, 1.0), (-0.25, 0.25), (0.7, 1)]
    orientation_bounds = [(0, 0.7071), (0, 0.7071), (0, 0.7071), (0, 1)]
    def __init__(self, is_train, is_dense=False):
        self.connect(is_train)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.is_dense = is_dense
        self.distance_threshold = 0.15

        self.planeId = None
        self.robot = None
        self.joints = []
        self.links = {}

        self.step_id = None
        self.object = None

        self.initial_joint_values = np.array([1, 0.0, 1.0])
        self.gripper_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        self.joint_values = None
        self.gripper_value = None
        self.dt = pybullet.getPhysicsEngineParameters()['fixedTimeStep']
        self.target_position = np.array([1, 0, 1])

    def step(self, action):
        self.step_id += 1
        self.joint_values += np.array(action[:3]) * 0.1
        self.joint_values = np.clip(self.joint_values, -1, 1)
        self.gripper_value = 1 if action[3] > 0 else -1
        # end effector points down, not up (in case useOrientation==1)
        target_pos = self.joint_values #self._rescale(self.joint_values, self.position_bounds)
        self.move_hand(target_pos, self.gripper_orientation, self.gripper_value)
        #wrist_1_joint_name = 'robot_wrist_1_joint'
        #wrist_1_joint_id = self.links[wrist_1_joint_name]
        #print(np.rad2deg(pybullet.getJointState(self.robot, wrist_1_joint_id)[0]))
        pybullet.stepSimulation()

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
           # Add recording logic here
        

        info = self.compute_info(action)
        return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), self.is_done(), info

    # def throw_v_p(self,action):
        
    #     self.step_id += 1
    #     print(self.step_id)
        
    #     # Define the target position and velocity for the end effector
    #     target_position = np.array([0.76, 0, 0.8])
    #     target_velocity = np.array([0, 0, 0.5])
        
    #     # Get the current position and orientation of the end effector
    #     current_position, current_orientation = pybullet.getLinkState(self.robot, self.links['robot_wrist_3_joint'])[:2]
        
    #     # Calculate the required velocity to reach the target position
    #     delta_position = target_position - np.array(current_position)
    #     required_velocity = delta_position / self.dt
        
    #     # Clip the required velocity to the target velocity
    #     required_velocity = np.clip(required_velocity, -target_velocity, target_velocity)
        
    #     # Set the joint motor control to achieve the required velocity
    #     for joint_id in self.links.values():
    #         pybullet.setJointMotorControl2(
    #             self.robot,
    #             joint_id,
    #             pybullet.VELOCITY_CONTROL,
    #             targetVelocity=required_velocity[joint_id % 3]
    #         )
    #         pybullet.stepSimulation()
        
    #     # Release the gripper at the end of the throw
    #     if self.step_id < 225:
    #         self.gripper_value = 10  # Fully close the gripper
    #     else:
    #         self.gripper_value = -1
        
    #     gripper_joint_name = 'rh_p12_rn'
    #     gripper_joint_id = self.links[gripper_joint_name]
    #     gripper_joint_info = self.joints[gripper_joint_id - 6]
        
    #     lower_limit, upper_limit = gripper_joint_info['jointLowerLimit'], gripper_joint_info['jointUpperLimit']
    #     target_position = (self.gripper_value + 1) / 2 * (upper_limit - lower_limit) + lower_limit
        
    #     pybullet.setJointMotorControl2(
    #         self.robot,
    #         gripper_joint_id,
    #         pybullet.POSITION_CONTROL,
    #         targetPosition=target_position,
    #     )
    #     pybullet.stepSimulation()
    #     object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
    #     info = self.compute_info(action)

    #     return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), self.is_done(), info
    def curl(self,action):
        self.step_id += 1
        step = self.step_id - 300
        wrist_1_joint_name = 'robot_wrist_1_joint'
        wrist_1_joint_id = self.links[wrist_1_joint_name]
        elbow_joint_name = 'robot_elbow_joint'
        elbow_joint_id = self.links[elbow_joint_name]
        shoulder_joint_name = "robot_shoulder_lift_joint"
        shoulder_joint_id = self.links[shoulder_joint_name]
        
    # Control the joint angle of the specified joints
        joints_to_control = [shoulder_joint_id, elbow_joint_id, wrist_1_joint_id]
        joint_angles = [np.rad2deg(pybullet.getJointState(self.robot, joint_id)[0]) for joint_id in joints_to_control]
        target_angles = np.deg2rad([-90, 135, 90])  # Assuming action is a list of target angles for each joint
        # total_steps = 100
        # for step in range(total_steps):
        for joint_id, target_angle in zip(joints_to_control, target_angles):
            current_position = pybullet.getJointState(self.robot, joint_id)[0]
            current_position_degrees = math.degrees(current_position)
            new_position_degrees = current_position_degrees + (math.degrees(target_angle) - current_position_degrees) * (step + 1) / 100
            pybullet.setJointMotorControl2(
                self.robot,
                joint_id,
                pybullet.POSITION_CONTROL,
                targetPosition=math.radians(new_position_degrees),
            )
        self.joint_values += np.array(action[:3]) * 0.1
        self.joint_values = np.clip(self.joint_values, -1, 1)
        self.gripper_value = 1 if action[3] > 0 else -1
        target_pos = self.joint_values
        # joint_poses = pybullet.calculateInverseKinematics(
        #     self.robot,
        #     self.links['rh_p12_rn'],  # End effector link
        #     target_pos,
        #     self.gripper_orientation,
        #     lowerLimits=[-3.141592653589793, -1.57, 2.355, -1.57, -3.141592653589793, -3.141592653589793],
        #     upperLimits=[3.141592653589793, -1.57, 2.355, -1.57, 3.141592653589793, 3.141592653589793],
        #     maxNumIterations=100,
        #     residualThreshold=.01
        #     )
        # # Get the joint in self.joints
        # # joint_id is the index of the joint in self.joints, which corresponds to the joint ID in pybullet
        # # joint_id = 0: robot_shoulder_pan_joint
        # # joint_id = 1: robot_shoulder_lift_joint
        # # joint_id = 2: robot_elbow_joint
        # # joint_id = 3: robot_wrist_1_joint
        # # joint_id = 4: robot_wrist_2_joint
        # # joint_id = 5: robot_wrist_3_joint
        # # Control UR10 arm joints
        # for joint_id in range(6):
        #     if joint_id not in [1, 2, 3]:
        #         pybullet.setJointMotorControl2(
        #             self.robot,
        #             self.joints[joint_id]['jointID'],
        #             pybullet.POSITION_CONTROL,
        #             targetPosition=joint_poses[joint_id],
        #         )
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[0]
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        pybullet.stepSimulation()
        info = self.compute_info(action) 
        return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), self.is_done(), info

    
    def throw(self, action,change):
        self.step_id += 1
        done =0 
        # Get the joint ID for the wrist 3 link
        wrist_1_joint_name = 'robot_wrist_1_joint'
        wrist_1_joint_id = self.links[wrist_1_joint_name]
        elbow_joint_name = 'robot_elbow_joint'
        elbow_joint_id = self.links[elbow_joint_name]
        shoulder_joint_name = "robot_shoulder_lift_joint"
        shoulder_joint_id = self.links[shoulder_joint_name]
        # Define the throwing motion parameters
        throw_duration = 1  # Number of simulation steps for the throw
        max_angle = np.pi  # Maximum angle for the throw (90 degrees)
        #angular_velocity = max_angle / throw_duration
        # Calculate the angular velocity needed for the throw
        # Perform the throwing motion
        for _ in range(throw_duration):
            # Move the wrist 3 joint
            pybullet.setJointMotorControl2(
                self.robot,
                wrist_1_joint_id,
                pybullet.VELOCITY_CONTROL,
                targetVelocity=-action
            )
            pybullet.setJointMotorControl2(
                self.robot,
                elbow_joint_id,
                pybullet.VELOCITY_CONTROL,
                targetVelocity=-action
            )
            pybullet.setJointMotorControl2(
                self.robot,
                shoulder_joint_id,
                pybullet.VELOCITY_CONTROL,
                targetVelocity=-action
            )
            pybullet.stepSimulation()
        
        # Release the gripper at the end of the throw
        joint = np.rad2deg(pybullet.getJointState(self.robot, wrist_1_joint_id)[0])
        if joint<75:
            self.gripper_value = -1 # Fully close the gripper
            object_position = pybullet.getBasePositionAndOrientation(self.object)[0]
        else:
            self.gripper_value = 10
        gripper_joint_name = 'rh_p12_rn'
        gripper_joint_id = self.links[gripper_joint_name]
        gripper_joint_info = self.joints[gripper_joint_id - 6]
        
        lower_limit, upper_limit = gripper_joint_info['jointLowerLimit'], gripper_joint_info['jointUpperLimit']
        target_position = (self.gripper_value + 1) / 2 * (upper_limit - lower_limit) + lower_limit
        
        pybullet.setJointMotorControl2(
            self.robot,
            gripper_joint_id,
            pybullet.POSITION_CONTROL,
            targetPosition=target_position,
        )
        pybullet.setJointMotorControl2(
            self.robot,
            12,
            pybullet.POSITION_CONTROL,
            targetPosition=target_position,
        )
        ### FOR STICKING OBJECT TO END EFFECTOR ###
        # constraint_id = p.createConstraint(parentBodyUniqueId=self.robot,
        #                            parentLinkIndex=self.links['rh_p12_rn'],
        #                            childBodyUniqueId=self.object,
        #                            childLinkIndex=-1,
        #                            jointType=p.JOINT_FIXED,
        #                            jointAxis=[0, 0, 0],
        #                            parentFramePosition=[0, 0, 0],
        #                            childFramePosition=[0, 0, 0])
        #if self.step_id==25:
            #pybullet.resetBaseVelocity(self.object, linearVelocity=(4,0,4))
            #p.removeConstraint(constraint_id)
        joint = np.rad2deg(pybullet.getJointState(self.robot, elbow_joint_id)[0])
        if joint <90:
            change=1
        if joint >135:
            change =0
        pybullet.stepSimulation()
        # Get the object's new position and orientation
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        # Compute the necessary information
        if object_pos[2] < 1 and self.gripper_value==-1:
            done=1
            # Create a DataFrame to store the action and object position
            data = pd.DataFrame([action] + list(object_pos))

            # Append the data to an existing Excel file or create a new one if it doesn't exist
            if os.path.exists('action_object_pos_log.xlsx'):
                existing_data = pd.read_excel('action_object_pos_log.xlsx')
                combined_data = pd.concat([existing_data, data])
                combined_data.to_excel('action_object_pos_log.xlsx', index=False, header=False)
            else:
                data.to_excel('action_object_pos_log.xlsx', index=False, header=False)
        info = self.compute_info(action)
        
        return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), done, info,change
    
    # def throw_hand(self, target_position, orientation, gripper_value):
    #     joint_poses = pybullet.calculateInverseKinematics(
    #         self.robot,
    #         self.links['rh_p12_rn'],  # End effector link
    #         target_position,
    #         orientation,
    #         maxNumIterations=100,
    #         residualThreshold=.01
    #     )
    #     joint_id = 6
    #     # Control UR10 arm joints
    #     # for joint_id in range(6):
    #     pybullet.setJointMotorControl2(
    #         self.robot,
    #         self.joints[joint_id]['jointID'],
    #         pybullet.POSITION_CONTROL,
    #         targetPosition=joint_poses[joint_id],
    #     )

    #     # Control Robotis RH-P12-RN gripper
    #     gripper_joint_name = 'rh_p12_rn'
    #     gripper_joint_id = self.links[gripper_joint_name]
    #     gripper_joint_info = self.joints[gripper_joint_id - 6]  # Subtract 6 because gripper joint comes after 6 arm joints
        
    #     # Map gripper_value from [-1, 1] to [lower_limit, upper_limit]
    #     lower_limit, upper_limit = gripper_joint_info['jointLowerLimit'], gripper_joint_info['jointUpperLimit']
    #     target_position = (gripper_value + 1) / 2 * (upper_limit - lower_limit) + lower_limit

    #     pybullet.setJointMotorControl2(
    #         self.robot,
    #         gripper_joint_id,
    #         pybullet.POSITION_CONTROL,
    #         targetPosition=target_position,
    #     )
    # def get_traj_from_ruckig(q0, q0_dot, qd, qd_dot, base0, based):
    #     inp = InputParameter(9)
    #     zeros2 = np.zeros(2)
    #     inp.current_position = np.concatenate((q0, base0))
    #     inp.current_velocity = np.concatenate((q0_dot, zeros2))
    #     inp.current_acceleration = np.zeros(9)

    #     inp.target_position = np.concatenate((qd, based))
    #     inp.target_velocity = np.concatenate((qd_dot, zeros2))
    #     inp.target_acceleration = np.zeros(9)

    #     inp.max_velocity = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 2.0, 2.0])
    #     inp.max_acceleration = np.array([15, 7.5, 10, 12.5, 15, 20, 20, 5.0, 5.0]) -1.0
    #     inp.max_jerk = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000, 1000, 1000]) - 100

    #     otg = Ruckig(9)
    #     trajectory = Trajectory(9)
    #     _ = otg.calculate(inp, trajectory)
    #     return trajectory

    # def get_full_throwing_config(robot):
    # """
    # Return full throwing configurations
    # :param robot:
    # :param q:
    # :param phi:
    # :param throw:
    # :return:
    # """
    #     r_throw = throw[0]
    #     z_throw = throw[1]
    #     r_dot = throw[2]
    #     z_dot = throw[3]

    #     # bullet fk
    #     controlled_joints = [6]
    #     # p.resetJointStatesMultiDof(robot, controlled_joints, [[q0_i] for q0_i in q])
    #     gripper_joint_name = 'rh_p12_rn'
    #     gripper_joint_id = self.links[gripper_joint_name]
    #     AE =p.getLinkState(robot, gripper_joint_id)[0]
    #     q = q.tolist()
    #     J, _ = p.calculateJacobian(robot, gripper_joint_id, [0, 0, 0], [0.1, 0.1], [0.0]*9, [0.0]*9)
    #     J = np.array(J)
    #     J = J[:,:7]

    #     throwing_angle = 45
    #     EB_dir = np.array([np.cos(throwing_angle), np.sin(throwing_angle)])

    #     J_xyz = J[:3, :]
    #     J_xyz_pinv = np.linalg.pinv(J_xyz)

    #     eef_velo = np.array([EB_dir[0]*r_dot, EB_dir[1]*r_dot, z_dot])
    #     q_dot = J_xyz_pinv @ eef_velo
    #     box_position = AE + np.array([-r_throw*EB_dir[0], -r_throw*EB_dir[1], -z_throw])

        # TODO: fix the gripper issue
        # from https://www.programcreek.com/python/example/122109/pybullet.getEulerFromQuaternion
        # gripperState = p.getLinkState(robot, 11)
        # gripperPos = gripperState[0]
        # gripperOrn = gripperState[1]
        # invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
        # eef_velo_dir_3d = eef_velo / np.linalg.norm(eef_velo)
        # tmp = AE + eef_velo_dir_3d
        # blockPosInGripper, _ = p.multiplyTransforms(invGripperPos, invGripperOrn, tmp, [0, 0, 0, 1])
        # velo_angle_in_eef = np.arctan2(blockPosInGripper[1], blockPosInGripper[0])

        # if (velo_angle_in_eef<0.5*math.pi) and (velo_angle_in_eef>-0.5*math.pi):
            # eef_angle_near = velo_angle_in_eef
        # elif velo_angle_in_eef>0.5*math.pi:
            # eef_angle_near = velo_angle_in_eef - math.pi
        # else:
            # eef_angle_near = velo_angle_in_eef + math.pi

        # q[-1] = eef_angle_near
        # return (q, phi, throw, q_dot, blockPosInGripper, eef_velo, AE)

    def throwing_physics():
        #ul = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) #need to change
        #ll = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) #need to change
        # q0 = 0.5*(u1+l1)
        # q0_dot = np.zeros(7)
        throw_config_full = get_full_throwing_config(self.robot)
        # traj_throw = get_traj_from_ruckig(q0=q0,q0_dot=q0_dot,qd=throw_config_full[0],base0=base0, based=-throw_config_full[-1][:-1])
        g = -9.81
        PANDA_BASE_HEIGHT = 0.5076438625
        # box_position = 2 # range
        # clid = p.connect(p.GUI)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=160, cameraPitch=-40, cameraTargetPosition=[0.75, -0.75, 0])

        # NOTE: need high frequency
        hz = 1000
        delta_t = 1.0 / hz
        # p.setGravity(0, 0, g)
        p.setTimeStep(delta_t)
        # p.setRealTimeSimulation(0)

        AE = 0.7
        EB = 2

        controlled_joints = [5]
        gripper_joints = [10, 12]
        numJoints = len(controlled_joints)
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robotEndEffectorIndex = 14
        if tt > plan_time - 1*delta_t:
            p.resetJointState(robotId, gripper_joints[0], 0.05)
            p.resetJointState(robotId, gripper_joints[1], 0.05)
        else:
            eef_state = p.getLinkState(robotId, robotEndEffectorIndex, computeLinkVelocity=1)
            p.resetBasePositionAndOrientation(soccerballId, eef_state[0], [0, 0, 0, 1])
            p.resetBaseVelocity(soccerballId, linearVelocity=eef_state[-2])
        #robotId = p.loadURDF("descriptions/rbkairos_description/robots/rbkairos_panda_hand.urdf", [-box_position[0], -box_position[1], 0], useFixedBase=True)

        #planeId = p.loadURDF("plane.urdf", [0, 0, 0.0])
        #soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.05)
        #boxId = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/box.urdf",
        #                [0, 0, PANDA_BASE_HEIGHT+box_position[2]],
        #                globalScaling=0.5)
        #p.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
        #                spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
        #p.changeDynamics(planeId, -1, restitution=0.9)
       # p.changeDynamics(robotId, gripper_joints[0], jointUpperLimit=100)
        #p.changeDynamics(robotId, gripper_joints[1], jointUpperLimit=100)

        # t0, tf = 0, trajectory.duration
        # plan_time = tf - t0
        # sample_t = np.arange(0, tf, delta_t)
        # n_steps = sample_t.shape[0]
        # traj_data = np.zeros([3, n_steps, 7])
        # base_traj_data = np.zeros([3, n_steps, 2])





    def reset(self):
        self._reset_world()

        x_position = 0.75
        y_position = 0.0
        #x_position = np.random.uniform(0,0.5)
        #y_position = np.random.uniform(0,0.5)
        # FIXME add rotation
        orientation = pybullet.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])

        #self.object = pybullet.loadURDF('cube_small.urdf', [x_position, y_position, 0.6], orientation, globalScaling=0.75)

        self.object = pybullet.loadURDF('models/ycb/apple.urdf', [x_position, y_position, 0.7], orientation, globalScaling=1)
        pybullet.stepSimulation()

        return self.compute_state()

    def start_log_video(self, filename):
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, filename)

    def stop_log_video(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    def render(self, mode='human'):
        pass

    def __del__(self):
        pybullet.disconnect()
    
    def simulation(self):
        pybullet.stepSimulation() 
   
    def compute_state(self):
        state = np.zeros(3 * 4 + 3 * 4 + 1)
        gripper_position, gripper_orientation, _, _, _, _, gripper_velocity, gripper_angular_velocity = \
            pybullet.getLinkState(self.robot, linkIndex=self.links['rh_p12_rn'], computeLinkVelocity=True)
        state[:3] = gripper_position
        state[3:6] = pybullet.getEulerFromQuaternion(gripper_orientation)
        state[6:9] = gripper_velocity
        state[9:12] = gripper_angular_velocity
        # Get the environment time step

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        object_velocity, object_angular_velocity = pybullet.getBaseVelocity(self.object)
        state[12:15] = np.asarray(object_pos) - gripper_position
        state[15:18] = pybullet.getEulerFromQuaternion(object_orient)
        state[18:21] = object_velocity
        state[21:24] = object_angular_velocity

        state[24] = self.compute_gripper_position()

        return {'observation': state, 'desired_goal': self.target_position, 'achieved_goal': object_pos}

    def compute_reward(self, achieved_goal, desired_goal, info):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        if self.is_dense:
            gripper_position, gripper_orientation, _, _, _, _, gripper_velocity, gripper_angular_velocity = \
                pybullet.getLinkState(self.robot, linkIndex=self.links['rh_p12_rn'],
                                      computeLinkVelocity=True)

            gripper_distance = np.linalg.norm(achieved_goal - np.asarray(gripper_position))
            return 1 - min(distance, 0.5) - gripper_distance
        else:
            return -(distance > self.distance_threshold).astype(np.float32)

    def is_done(self):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        return self.step_id == 400 #or np.linalg.norm(object_pos - np.array([1.0, 0.0, 0.6])) > 1.0

    def compute_info(self, last_action):
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        distance = np.linalg.norm(object_pos - self.target_position)
        return {
            'is_success': distance < self.distance_threshold,
            'gripper_pos': self.compute_gripper_position(),
            'last_action': last_action
        }


    def connect(self, is_train):
        if is_train:
            pybullet.connect(pybullet.DIRECT)
        else:
            pybullet.connect(pybullet.GUI)

    def _rescale(self, values, bounds):
        result = np.zeros_like(values)
        for i, (value, (lower_bound, upper_bound)) in enumerate(zip(values, bounds)):
            result[i] = (value + 1) / 2 * (upper_bound - lower_bound) + lower_bound
        return result

    def call_camera(self):
        width = 320
        height = 240
        fov = 10.0
        aspect = width / height
        # The view matrix is not explicitly defined in the original code. It seems to be a hardcoded matrix for a specific camera view.
        camera_position = [1, 0, 1.5]       # Camera position [X, Y, Z]
        camera_target_position = [1, 0, 0]  # The point the camera is looking at
        camera_up_vector = [0, 1, 0]        # The up direction for the camera

# Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector
        )
        # Assuming the hardcoded matrix is correct for the desired view, we will keep it as is.
        #view_matrix = [[1, 0, 0, 0],[0, 0.6428, -0.766, 0],[0, 0.766, 0.6428, 0],[0, 0, -1, 1]]
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60.0,            # Field of view
            aspect=1.0,          # Aspect ratio (width/height of image)
            nearVal=0.1,         # Near clipping plane
            farVal=100.0         # Far clipping plane
        )
        #projection_matrix = [[1, 0, 0, 0], [0, 0.6428, -0.766, 0], [0, 0.766, 0.6428, 0], [0, 0, -5, 1]]
        images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        depth_buffer_opengl = np.reshape(images[3], [width, height])
    
    def move_hand(self, target_position, orientation, gripper_value):
        # Calculate inverse kinematics for the UR10 arm
       
        joint_poses = pybullet.calculateInverseKinematics(
            self.robot,
            self.links['rh_p12_rn'],  # End effector link
            target_position,
            orientation,
            maxNumIterations=100,
            residualThreshold=.01
        )

        # Control UR10 arm joints
        for joint_id in range(6):
            pybullet.setJointMotorControl2(
                self.robot,
                self.joints[joint_id]['jointID'],
                pybullet.POSITION_CONTROL,
                targetPosition=joint_poses[joint_id],
            )

        # Control Robotis RH-P12-RN gripper
        gripper_joint_name = 'rh_p12_rn'
        gripper_joint_id = self.links[gripper_joint_name]
        gripper_joint_info = self.joints[gripper_joint_id - 6]  # Subtract 6 because gripper joint comes after 6 arm joints
        
        # Map gripper_value from [-1, 1] to [lower_limit, upper_limit]
        lower_limit, upper_limit = gripper_joint_info['jointLowerLimit'], gripper_joint_info['jointUpperLimit']
        target_position = (gripper_value + 1) / 2 * (upper_limit - lower_limit) + lower_limit 
        pybullet.setJointMotorControl2(
            self.robot,
            gripper_joint_id,
            pybullet.POSITION_CONTROL,
            targetPosition=target_position,
        )
        pybullet.setJointMotorControl2(
            self.robot,
            12,
            pybullet.POSITION_CONTROL,
            targetPosition=target_position,
        )


    def compute_gripper_position(self):
        """
        This function computes the position of the gripper based on the UR10 robot with Robotis RH-P12-RN gripper.

        It does the following:
        1. Retrieves the state of the single prismatic joint that controls the gripper.
        2. Normalizes the joint position to a range of [-1, 1] based on its limits.

        The returned value represents the gripper position, where:
        - A value close to -1 indicates a fully open gripper.
        - A value close to 1 indicates a fully closed gripper.
        - Values in between represent partial closure states.

        Returns:
            float: The normalized position of the gripper.
        """
        # The UR10 with Robotis RH-P12-RN gripper uses a single prismatic joint named 'rh_p12_rn'
        gripper_joint_name = 'rh_p12_rn'
        gripper_joint_id = self.links[gripper_joint_name]
        
        data = pybullet.getJointState(self.robot, gripper_joint_id)
        position = data[0]  # Current position of the joint
        
        joint_info = self.joints[gripper_joint_id - 6]  # Subtract 6 because gripper joint comes after 6 arm joints
        lower_bound, upper_bound = joint_info['jointLowerLimit'], joint_info['jointUpperLimit']
        
        # Normalize the position to [-1, 1]
        normalized_position = (position - lower_bound) / (upper_bound - lower_bound) * 2 - 1
        
        return normalized_position
    def load_model(rel_path, pose=None, **kwargs):
        """Load a model from a relative path and set its pose if given."""
        model_dir = os.path.abspath(os.path.dirname(__file__))
        abs_path = os.path.join(model_dir, "models", rel_path)
        # If the file does not exist in default models folder,
        # use the relative path directly
        if not os.path.isfile(abs_path):
            abs_path = rel_path

        body = load_pybullet(abs_path, **kwargs)
        if pose is not None:
            set_pose(body, pose)
        return body

    def _reset_world(self):
        pybullet.resetSimulation()
        # interactive_camera_placement()        
        pybullet.setGravity(0, 0, -9.8)
        self.planeId = pybullet.loadURDF('plane.urdf')
        robot_position = [0, 0, 0.7]
        
        robot_orientation = pybullet.getQuaternionFromEuler([0, 0, 0 ])
        self.robot = pybullet.loadURDF('models/robots/ur10_robotis_d435.urdf', robot_position,robot_orientation, globalScaling=1)
        num_joints = pybullet.getNumJoints(self.robot)
        joints=get_movable_joints(self.robot)
        width = 320
        height = 240
        fov = 10.0
        aspect = width / height
        view_matrix = [[1, 0, 0, 0],[0, 0.6428, -0.766, 0],[0, 0.766, 0.6428, 0],[0, 0, -5, 1]]
        projection_matrix=[[1,0,0,0],[0,0.6428,-0.766,0],[0,0.766,0.6428,0],[0,0,-5,1]]
        images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        depth_buffer_opengl = np.reshape(images[3], [width, height])
        # depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
        # set_joint_positions(ur10, joints, [0, -1.7, 2, -1.87, -1.57, 0, 0, 0])
     

        joint_type = ['REVOLUTE', 'PRISMATIC', 'SPHERICAL', 'PLANAR', 'FIXED']
        self.joints = []
        self.links = {}

        for joint_id in range(pybullet.getNumJoints(self.robot)):
            info = pybullet.getJointInfo(self.robot, joint_id)
            data = {
                'jointID': info[0],
                'jointName': info[1].decode('utf-8'),
                'jointType': joint_type[info[2]],
                'jointLowerLimit': info[8],
                'jointUpperLimit': info[9],
                'jointMaxForce': info[10],
                'jointMaxVelocity': info[11]
            }
            if data['jointType'] != 'FIXED':
                self.joints.append(data)
                self.links[data['jointName']] = joint_id
        self.step_id = 0
        self.object = None

        self.initial_joint_values = np.array([1.0, 0.0, 1.0])

        self.initial_joint_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        self.gripper_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        self.joint_values = self.initial_joint_values
        self.joint_orientation = self.initial_joint_orientation
        self.gripper_value = -1

        # pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0])
        floor =  pybullet.loadURDF("plane.urdf")
        table1 = pybullet.loadURDF("models/furniture/lab_table.urdf", [0.1, 0, 0])
        table2 = pybullet.loadURDF("models/furniture/lab_table.urdf", [0.6, 0, 0])
        # cabinet = pybullet.loadURDF("models/furniture/cabinet.urdf", [0.7, 0.65, 0.7])
        # drawer = pybullet.loadURDF("models/furniture/drawer.urdf",[0.7, -0.65, 0.7], pybullet.getQuaternionFromEuler([0, 0, 3.14159]))
        self.position_bounds = [(0.5, 1.0), (-0.25, 0.25), (0.7, 1)]
        self.orientation_bounds = [(0, 0.7071), (0, 0.7071), (0, 0.7071), (0, 1)]
        #peg_box=pybullet.loadURDF("C:\Users\mark1\Downloads\bullet3\data\mar.urdf")
        #set_pose(peg_box, Pose(Point(0.4,0.5, stable_z(peg_box, table1))))
       
        self.target_position = np.array([1, 0, 1])