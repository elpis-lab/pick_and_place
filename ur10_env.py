from __future__ import print_function
import pybullet
import pybullet_data
from PIL import Image
import numpy as np
import gym
import numpy as np
import math
import csv
import os
import pybullet as p
import time
import pandas as pd
import random
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

    def reset_robot(self):
        
        for joint_index, joint_position in enumerate(self.initial_joint_positions):
            p.resetJointState(self.robot, joint_index, joint_position)
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'], computeLinkVelocity=False)[0]
        gripper_value = -1
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
        pybullet.stepSimulation()

    def step(self, action):
        self.step_id += 1
        self.joint_values += np.array(action[:3]) * 0.1
        self.joint_values = np.clip(self.joint_values, -1, 1)
        self.gripper_value = 1 if action[3] > 0 else -1
        # end effector points down, not up (in case useOrientation==1)
        target_pos = self.joint_values #self._rescale(self.joint_values, self.position_bounds)
        self.move_hand(target_pos, self.gripper_orientation, self.gripper_value)
        pybullet.stepSimulation()

        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
           # Add recording logic here
        

        info = self.compute_info(action)
        return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), self.is_done(), info


    def pick(self,target):
        self.step_id += 1
        end_effector_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'], computeLinkVelocity=False)[0]
        # object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        if self.step_id<200:
            target_pos = end_effector_position + (target[:3] - end_effector_position)*(self.step_id/200)
        else:
            target_pos = end_effector_position + (target[:3] - end_effector_position)*(self.step_id/250)

        joint_poses = pybullet.calculateInverseKinematics(
            self.robot,
            self.links['rh_p12_rn'],  # End effector link
            target_pos,
            self.gripper_orientation,
            maxNumIterations=200,
            residualThreshold=.1
        )
        # Control UR10 arm joints
        for joint_id in range(6):
            pybullet.setJointMotorControl2(
                self.robot,
                self.joints[joint_id]['jointID'],
                pybullet.POSITION_CONTROL,
                targetPosition=joint_poses[joint_id],
            )
        p.stepSimulation()
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'], computeLinkVelocity=False)[0]
        gripper_value = target[3]
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
        pybullet.stepSimulation()

    def curl(self,action):
        
        self.step_id += 1
        step = self.step_id - 300
        wrist_3_joint_name = 'robot_wrist_3_joint'
        
        wrist_3_joint_id = self.links[wrist_3_joint_name]
        wrist_1_joint_name = 'robot_wrist_1_joint'
        wrist_1_joint_id = self.links[wrist_1_joint_name]
        elbow_joint_name = 'robot_elbow_joint'
        elbow_joint_id = self.links[elbow_joint_name]
        shoulder_joint_name = "robot_shoulder_lift_joint"
        shoulder_joint_id = self.links[shoulder_joint_name]
        shoulder_pan_joint_name =  "robot_shoulder_pan_joint"
        shoulder_pan_joint_id = self.links[shoulder_pan_joint_name]
        # Get the joint positions using self.links
    # Control the joint angle of the specified joints
        joints_to_control = [shoulder_pan_joint_id,shoulder_joint_id, elbow_joint_id, wrist_1_joint_id,wrist_3_joint_id]
        joint_angles = [np.rad2deg(pybullet.getJointState(self.robot, joint_id)[0]) for joint_id in joints_to_control]
        target_angles = np.deg2rad([90,-90, 135, 90, 90])  # Assuming action is a list of target angles for each joint
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
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[0]
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        pybullet.stepSimulation()
        info = self.compute_info(action) 
        return self.compute_state(), self.compute_reward(object_pos, self.target_position, info), self.is_done(), info

    
    # def dynamics(self):
    #     print("bye")
    #     #start_position = np.array([x_start, y_start, z_start])  # Replace with your start position coordinates
    #     duration = 2.0  # in seconds
    #     num_steps = 100  # More steps mean smoother motion
    #     target_position = np.array([-0.155, 0.508, 1.149])  # Replace with your target position coordinates
    #     # Time between each step
    #     dt = duration / num_steps

    #     start_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'], computeLinkVelocity=False)[0]
    #     # Create the time steps for interpolation
    #     time_stamps = np.linspace(0, duration, num_steps)
    #     positions = [start_position + (target_position - start_position) * (t / duration) for t in time_stamps]
    #     return positions
    #     # Control loop to move the end effector
    #         # Solve for joint positions using IK
    # def throw_tim(self,positions,step):
    #     #gripper_orientation = [0,0,np.pi/4]
    #     joint_poses = pybullet.calculateInverseKinematics(
    #         self.robot,
    #         self.links['rh_p12_rn'],  # End effector link
    #         positions[step],
    #         #gripper_orientation,
    #         maxNumIterations=200,
    #         residualThreshold=.1
    #     )
    #     # Apply the joint positions
    #     for i, joint_angle in enumerate(joint_poses):
    #         p.setJointMotorControl2(
    #             bodyIndex=self.robot,
    #             jointIndex=i,
    #             controlMode=p.POSITION_CONTROL,
    #             targetPosition=joint_angle
    #         )
    #     gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[0]
    #     # Step simulation and wait
    #     p.stepSimulation()
    #     time.sleep(0.02)
    def throw_real(self, action,grip_val,step):
        #end_effector_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'], computeLinkVelocity=False)[0]
        # object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        wrist_3_joint_id = self.links[wrist_3_joint_name]
        wrist_1_joint_name = 'robot_wrist_1_joint'
        wrist_1_joint_id = self.links[wrist_1_joint_name]
        elbow_joint_name = 'robot_elbow_joint'
        elbow_joint_id = self.links[elbow_joint_name]

        joints_to_control = [elbow_joint_id, wrist_1_joint_id]
        joint_angles = [np.rad2deg(pybullet.getJointState(self.robot, joint_id)[0]) for joint_id in joints_to_control]
        target_angles = np.deg2rad([-45,])  # Assuming action is a list of target angles for each joint


        p.stepSimulation()
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'], computeLinkVelocity=False)[0]
        gripper_value = target[3]
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
        pybullet.stepSimulation() 
    def throw(self, action,grip_val):
        self.step_id += 1
        done=0
        # Get the joint ID for the wrist 3 link
        wrist_1_joint_name = 'robot_wrist_1_joint'
        wrist_1_joint_id = self.links[wrist_1_joint_name]
        elbow_joint_name = 'robot_elbow_joint'
        elbow_joint_id = self.links[elbow_joint_name]
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[0]
        throw_duration = 1  # Number of simulation steps for the throw
        max_angle = np.pi  # Maximum angle for the throw (90 degrees)
        object_vel = [1.35,0,1.35]
        # Calculate the end effector linear velocity using joint velocities
        # Calculate the Jacobian matrix for the robot
        joint_velocities = [pybullet.getJointState(self.robot, joint_id)[1] for joint_id in self.links.values()]
        #joint_velocities = [pybullet.getJointState(self.robot, joint_id)[1] for joint_id in [wrist_1_joint_id, elbow_joint_id]]
        joint_positions = [pybullet.getJointState(self.robot, joint_id)[0] for joint_id in self.links.values()]
        #joint_positions = [pybullet.getJointState(self.robot, joint_id)[0] for joint_id in [wrist_1_joint_id, elbow_joint_id]]
        self.jacobian = pybullet.calculateJacobian(
            self.robot,
            self.links['rh_p12_rn'],  # End effector link
            [0,0,0],
            joint_positions,  # Position of the end effector
            joint_velocities,  # Orientation of the end effector (not used for linear velocity)
            [0, 0, 0, 0, 0, 0, 0, 0],  # Angular velocity of the joints (not used for linear velocity)
        )
        #joint_velocities = [pybullet.getJointState(self.robot, joint_id)[1] for joint_id in [wrist_1_joint_id, elbow_joint_id]]
        end_effector_linear_velocity = np.dot(self.jacobian, joint_velocities)
        (f"End effector linear velocity: {end_effector_linear_velocity}")
        print# object_pos = [pybullet.getJointState(self.robot, joint["jointID"])[0] for joint in (self.joints)]
        # theta_dot = [0,0,np.pi,np.pi,0,0,0,0]
        # jacobian = pybullet.calculateJacobian(self.robot, self.links['rh_p12_rn'], [0, 0, 0], object_pos, theta_dot, [0,0,0,0,0,0,0,0])
        
        if grip_val==10:
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
            gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[0]
            pybullet.stepSimulation()
            # This is a quaternion
            # The gripper orientation is with respect to the robot base orientation
            gripper_orientation = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[1]
            gripper_orientation_euler = pybullet.getEulerFromQuaternion(gripper_orientation)
            #print(gripper_orientation_euler)
            object_orientation = pybullet.getBasePositionAndOrientation(self.object)[1]
            object_orientation_euler = pybullet.getEulerFromQuaternion(object_orientation)
            print(object_orientation_euler)
            object_vel = pybullet.getBaseVelocity(self.object)[0]
            print(object_vel)
            if object_orientation_euler[1] <= -0.78:
                grip_val = -1 # Fully open the gripper
                # Extract the joint velocity  # get the angular velocity
                gripper_orientation = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[1]
                gripper_orientation_euler = pybullet.getEulerFromQuaternion(gripper_orientation)
                print(gripper_orientation_euler)
                object_vel = pybullet.getBaseVelocity(self.object)[0]
                print(object_vel)
                pybullet.resetBaseVelocity(self.object, linearVelocity=(object_vel[0], v_release, v_release))
                object_pos = [pybullet.getJointState(self.robot, joint["jointID"])[0] for joint in (self.joints)]
                theta_dot = [0,0,-action,-action,0,0,0,0]
                # jacobian = pybullet.calculateJacobian(self.robot, self.links['rh_p12_rn'], [0, 0, 0], object_pos, theta_dot, [0,0,0,0,0,0,0,0])
                # linear_velocity_jacobian = jacobian[0][:3]
                # vel = np.dot(linear_velocity_jacobian,theta_dot)
                done = 1
                #print(f"Gripper velocity: {vel}")
                #object_position = pybullet.getBasePositionAndOrientation(self.object)[0]
                gripper_orientation = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[1]
                gripper_orientation_euler = pybullet.getEulerFromQuaternion(gripper_orientation)
                gripper_joint_name = 'rh_p12_rn'
                gripper_joint_id = self.links[gripper_joint_name]
                gripper_joint_info = self.joints[gripper_joint_id - 6]
                
                lower_limit, upper_limit = gripper_joint_info['jointLowerLimit'], gripper_joint_info['jointUpperLimit']
                target_position = (grip_val + 1) / 2 * (upper_limit - lower_limit) + lower_limit
                
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
                # Set the velocity to the object
                #pybullet.resetBaseVelocity(self.object, linearVelocity=(1.35,0,1.35))
                pybullet.stepSimulation()
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)

        info = self.compute_info(action)
        
        return object_vel, gripper_position,self.compute_reward(object_pos, self.target_position, info), done
    
    def step_sim(self, vel, gripper_position, action, done):
        # Get the object's new position and orientation
        object_pos, object_orient = pybullet.getBasePositionAndOrientation(self.object)
        #object_vel = pybullet.getBaseVelocity(self.object)
        p_f = [0,0,0]
        Range = [0,0]
        if object_pos[2]<=0.75:
            done = 2
            #R_x =object_pos[0] - gripper_position[0]
            #R_y = object_pos[1] - gripper_position[1]
            z = gripper_position[2]
            x = gripper_position[0]
            y = gripper_position[1]
            o_x = object_pos[0]
            o_y = object_pos[1]
            p_f = [x,y,z]
            Range = [o_x,o_y]
                # Save R_x, R_y and the velocity vel in a CSV file
            with open('benchmark_5.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([x,y,o_x,o_y,vel[0],vel[1],vel[2],z]) 
        wrist_1_joint_name = 'robot_wrist_1_joint'
        wrist_1_joint_id = self.links[wrist_1_joint_name]
        elbow_joint_name = 'robot_elbow_joint'
        elbow_joint_id = self.links[elbow_joint_name]
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
        pybullet.stepSimulation()
        return vel,p_f,Range, done

    def reset(self,episode):
        rgb_d = self._reset_world(episode)
        #self.object = pybullet.loadURDF('models/ycb/banana.urdf', [x_position, y_position, 0.7], orientation, globalScaling=1)
        pybullet.stepSimulation()

        return self.compute_state(),rgb_d

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
        object_pos = (object_pos[0], object_pos[1], 0.75)
        object_velocity, object_angular_velocity = pybullet.getBaseVelocity(self.object)
        state[12:15] = np.asarray(object_pos) - gripper_position
        state[15:18] = pybullet.getEulerFromQuaternion(object_orient)
        state[18:21] = object_pos
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

    def call_camera(self,episode):
        width = 720
        height = 1280
        fov = 60.0
        aspect = width / height
        # The view matrix is not explicitly defined in the original code. It seems to be a hardcoded matrix for a specific camera view.
        camera_position = [0, 0.7, 1]       # Camera position [X, Y, Z]
        camera_target_position = [0, 0.7, 0.7]  # The point the camera is looking at
        camera_up_vector = [1, 0, 0]        # The up direction for the camera
       # The up direction for the camera
        # Rotate camera by 90 degrees
        angle = math.radians(90)
        distance = 1.0                    # Distance from the camera to the target

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector,
            #cameraForwardVector=[0, 0, -1],  # Assuming the camera is facing downwards
            #cameraRightVector=[1, 0, 0]  # Assuming the camera is facing to the right
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90.0,            # Field of view
            aspect=1280/720,          # Aspect ratio (width/height of image)
            nearVal=0.1,         # Near clipping plane
            farVal=100.0         # Far clipping plane
        )
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

    def gripper_position(self):
        gripper_position = pybullet.getLinkState(self.robot, self.links['rh_p12_rn'])[0]
        return gripper_position
    
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


    
    def _reset_world(self,episode):
        pybullet.resetSimulation()
        # interactive_camera_placement()        
        pybullet.setGravity(0, 0, -9.8)
        self.planeId = pybullet.loadURDF('plane.urdf')
        robot_position = [0, 0, 0.7]
        
        robot_orientation = pybullet.getQuaternionFromEuler([0, 0, 0 ])
        self.robot = pybullet.loadURDF('models/robots/ur10_robotis_d435.urdf', robot_position,robot_orientation, globalScaling=1)
        num_joints = pybullet.getNumJoints(self.robot)
        joints=get_movable_joints(self.robot)
        
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
        self.initial_joint_positions = []
        num_joints = p.getNumJoints(self.robot)
        for joint_index in range(num_joints):
            joint_state = p.getJointState(self.robot, joint_index)
            self.initial_joint_positions.append(joint_state[0]) 
        self.initial_joint_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, 0])
        self.gripper_orientation = pybullet.getQuaternionFromEuler([np.pi, 0, np.pi/2])
        self.joint_values = self.initial_joint_values
        self.joint_orientation = self.initial_joint_orientation
        self.gripper_value = -1

        # pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0])
        floor =  pybullet.loadURDF("plane.urdf")
        table1 = pybullet.loadURDF("models/furniture/lab_table.urdf", [0, 0.1, 0],pybullet.getQuaternionFromEuler([0,0,np.pi/2]))
        table2 = pybullet.loadURDF("models/furniture/lab_table.urdf", [0, 0.6, 0],pybullet.getQuaternionFromEuler([0,0,np.pi/2]))
        #table3 = pybullet.loadURDF("models/furniture/lab_table.urdf", [0, 1.1, 0],pybullet.getQuaternionFromEuler([0,0,np.pi/2]))
        # box = pybullet.loadURDF("models/furniture/box.urdf",[0,1,0.75])
        # box = pybullet.loadURDF("models/furniture/box.urdf",[-0.1,1.1,0.75],pybullet.getQuaternionFromEuler([0,0,np.pi/2]))
        # box = pybullet.loadURDF("models/furniture/box.urdf",[0,0.9,0.75])
        #box = pybullet.loadURDF("models/furniture/box1.urdf",[-0.1,1.1,0.75],pybullet.getQuaternionFromEuler([0,np.pi/2,0]))
        
        #box = pybullet.loadSTL("Part33.STL",[-0.1,1.1,0.75],pybullet.getQuaternionFromEuler([0,0,np.pi/2]))
        x_position_apple = 0
        y_position_apple = 0.5
        x_position_banana = 0
        y_position_banana = 0.6
        x_position_driver = 0.1
        y_position_driver = 0.5
        #x_position = np.random.uniform(0,0.5)
        #y_position = np.random.uniform(0,0.5)
        # FIXME add rotation
        self.strawberry = pybullet.loadURDF("models/ycb/strawberry.urdf", [x_position_apple, y_position_apple, 0.75], pybullet.getQuaternionFromEuler([0, 0, np.pi/2]))
        self.peach = pybullet.loadURDF("models/ycb/peach.urdf", [x_position_banana, y_position_banana, 0.75], pybullet.getQuaternionFromEuler([0, 0, np.pi/2]))
        self.lemon = pybullet.loadURDF("models/ycb/lemon.urdf", [x_position_driver, y_position_driver, 0.75], pybullet.getQuaternionFromEuler([0, 0, np.pi/2]))
        orientation = pybullet.getQuaternionFromEuler([0, 0, np.pi/2])
        self.object = self.strawberry
        #self.object = pybullet.loadURDF('cube_small.urdf', [x_position, y_position, 0.6], orientation, globalScaling=0.75)
        #object_mass = 100.0  # Define the mass of the object
        
        #pybullet.changeDynamics(self.object, -1, mass=object_mass)  # Add the mass to the simulation
        
        # cabinet = pybullet.loadURDF("models/furniture/cabinet.urdf", [0.7, 0.65, 0.7])
        # drawer = pybullet.loadURDF("models/furniture/drawer.urdf",[0.7, -0.65, 0.7], pybullet.getQuaternionFromEuler([0, 0, 3.14159]))
        self.position_bounds = [(0.5, 1.0), (-0.25, 0.25), (0.7, 1)]
        self.orientation_bounds = [(0, 0.7071), (0, 0.7071), (0, 0.7071), (0, 1)]
        #peg_box=pybullet.loadURDF("C:\Users\mark1\Downloads\bullet3\data\mar.urdf")
        #set_pose(peg_box, Pose(Point(0.4,0.5, stable_z(peg_box, table1))))
       
        self.target_position = np.array([1, 0, 1])
        width = 1280
        height = 720
        fov = 90.0
        aspect = width / height
        # The view matrix is not explicitly defined in the original code. It seems to be a hardcoded matrix for a specific camera view.
        camera_position = [0, 0.7, 1]       # Camera position [X, Y, Z]
        camera_target_position = [0, 0.7, 0.7]  # The point the camera is looking at
        camera_up_vector = [1, 0, 0]        # The up direction for the camera
       # The up direction for the camera
        # Rotate camera by 90 degrees
        angle = math.radians(90)
        distance = 1.0                    # Distance from the camera to the target

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=camera_up_vector,
            #cameraForwardVector=[0, 0, -1],  # Assuming the camera is facing downwards
            #cameraRightVector=[1, 0, 0]  # Assuming the camera is facing to the right
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,            # Field of view
            aspect=aspect,          # Aspect ratio (width/height of image)
            nearVal=0.1,         # Near clipping plane
            farVal=100.0         # Far clipping plane
        )
        images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        import cv2
        rgb_image = images[2]
        rgb_array = np.array(rgb_image).reshape(height, width, 4)
# Extract only the RGB channels (ignore alpha channel)
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)
        cv2.imwrite(f'RGB_image_{episode}.jpg', cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))


        # Save the image using PIL
                #cv2.imwrite(f'rgb_image_{episode}.jpg', np.array(images, dtype=np.uint8))
        # Process and save depth image (normalized for better visualization)
        depth_buffer = np.array(images[3]).reshape((height, width))
        depth_img = Image.fromarray((depth_buffer * 255).astype(np.uint8), mode='L')  # Grayscale depth
        depth_img.save(f'depth_image_{episode}.jpg')
        rgb_d = np.dstack((rgb_array, depth_img))
        print(f"Saved images for episode {episode}")
        return rgb_d