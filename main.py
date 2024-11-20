#from gym.wrappers import FlattenObservation
import time
#from stable_baselines.common.policies import MlpPolicy
#from stable_baselines import PPO2
#from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
from tqdm import trange
import argparse
from ur10_env import UR10


class HardcodedAgent(object):
    def __init__(self, position_bounds, orientation_bounds):
        self.step = 0
        self.position_bounds = position_bounds
        self.orientation_bounds = orientation_bounds
    #Harcoded Pick and Place Policy
    def act(self, state):
        state = state['observation']
        self.step += 1
        
        gripper_pos = state[:3]
        gripper_orientation = state[3:7]
        delta_pos = state[12:15]
        object_pos = state[18:21]
        # object_pos = gripper_pos + delta_pos
        target_pos = np.zeros(4)
        target_orient = np.zeros(5)
        if self.step<200:
            target_pos[:3] = object_pos
            target_pos[0] -= 0.05 
            target_pos[2] = object_pos[2]+ 0.2
            target_pos[3] = -1
        elif self.step<250:
            target_pos[:3] = object_pos
            target_pos[0] -= 0.05
            target_pos[2] = object_pos[2]+ 0.05
            target_pos[3] = -1
        else:
            target_pos[:3] = object_pos
            target_pos[2] = object_pos[2]+ 0.05
            target_pos[3] = 10
        return target_pos
        
    def reset(self):
        self.step = 0

    def _rescale(self, values, bounds):
        result = np.zeros_like(values)
        for i, (value, (lower_bound, upper_bound)) in enumerate(zip(values, bounds)):
            result[i] = value / (upper_bound - lower_bound)
        return result

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state):
        return self.action_space.sample()

    def reset(self):
        pass


class RLAgent(object):
    def __init__(self, model):
        self.model = model

    def act(self, state):
        return self.model.predict([state])[0][0]

    def reset(self):
        pass


def evaluate(policy, env, episode,objects, viz=True):
    env.reset_robot()
    policy.reset()
    state = env.compute_state()
    #env.reset_robot_to_base_position()
    while True:
        if env.step_id <= 300:
            #print(env.step_id)
            env.call_camera(episode)
            pos = policy.act(state)
            #stte, rew, done, info = env.step(pos)
            env.pick(pos)
            rew = 1
            done =0
        else:
            current_pos = policy.act(state)
            env.call_camera(episode)
            state,rew,done,info = env.curl(pos)
        if viz:
            time.sleep(0.001)
        if done:
            break
                                        #Throwing Starts HERE"
    # # Calculate the angular velocity in radians per second
    # # throw_duration_seconds = 1  # Duration of throw in seconds
    # # max_angle_radians = np.random.uniform(np.pi/2,np.pi) # Maximum angle for the throw (180 degrees)
    # # #max_angle_radians = np.pi
    # # angular_velocity_rad_per_sec = max_angle_radians / throw_duration_seconds
    # # print(angular_velocity_rad_per_sec)
    # # Calculate the number of simulation steps
    # throw_duration_seconds = 0.6
    # simulation_steps = int(throw_duration_seconds / env.dt)  # Assuming env.dt is the simulation time step
    # # Perform the throwing motion
    
    # grip_val=10
    # done= 0
    # vel = [0,0,0]
    # gripper_position = [0,0,0]
    # #positions = env.dynamics()
    # step = 0
    # #while step<100:
    # while True:
    #     # Convert angular velocity from rad/sec to rad/step
    #     env.call_camera(episode)
    #     # Move the wrist joint
    #     if done==0:
    #         #print("hi")
    #         #env.throw_tim(positions,step)
    #         #step= step+1
    #         vel,gripper_position, rew, done = env.throw(angular_velocity_rad_per_sec,grip_val)
    #     elif done==1:
    #         vel, p_f, Range, done = env.step_sim(vel, gripper_position,angular_velocity_rad_per_sec,done)
    #     else:
    #         time.sleep(2)
    #         break
    #     if viz:
    #         time.sleep(0)
    # Calculate the angular velocity in radians per second
    # throw_duration_seconds = 1  # Duration of throw in seconds
    # max_angle_radians = np.random.uniform(np.pi/2,np.pi) # Maximum angle for the throw (180 degrees)
    # #max_angle_radians = np.pi
    # angular_velocity_rad_per_sec = max_angle_radians / throw_duration_seconds
    # print(angular_velocity_rad_per_sec)
    # Calculate the number of simulation steps
    throw_duration_seconds = 0.6
    simulation_steps = int(throw_duration_seconds / env.dt)  # Assuming env.dt is the simulation time step
    # Perform the throwing motion
    
    grip_val=10
    done= 0
    vel = [0,0,0]
    gripper_position = [0,0,0]
    #positions = env.dynamics()
    step = 0
    #while step<100:
    while True:
        # Convert angular velocity from rad/sec to rad/step
        env.call_camera(episode)
        # Move the wrist joint
        if done==0:
            #print("hi")
            #env.throw_tim(positions,step)
            #step= step+1
            vel,gripper_position, rew, done = env.throw_real(angular_velocity_rad_per_sec,grip_val)
        elif done==1:
            vel, p_f, Range, done = env.step_sim(vel, gripper_position,angular_velocity_rad_per_sec,done)
        else:
            time.sleep(2)
            break
        if viz:
            time.sleep(0)
    objects = objects-1
        # Ensure the gripper is fully opened after the throw
        # env.open_gripper()
        # state, rew, done, info = env.throw(pos)
    return vel, p_f, Range

def make_video(policy, env, out='video.mp4'):
    state = env.reset()
    policy.reset()
    env.start_log_video(out)
    while True:
        state, rew, done, info = env.step(policy.act(state))
        time.sleep(0.05)
        if done:
            break
    env.stop_log_video()
    env.close()


def train_ppo(nsteps):
    train_env = SubprocVecEnv([lambda: FlattenObservation(UR10(is_train=True, is_dense=True))] * 8)
    model = PPO2(MlpPolicy, train_env,
                 verbose=1, tensorboard_log='log',
                 policy_kwargs={'layers': [256, 256, 256]},
                 )
    model.learn(total_timesteps=int(nsteps))
    model.save('ppo_model')

# def ResidualPlan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, choices=['ppo', 'script', 'random'])
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'viz', 'video'])
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.agent == 'ppo', f'{args.agent} agent is not trainable'
        train_ppo(2e5)
    else:
        if args.agent == 'ppo':
            env = FlattenObservation(UR10(is_train=args.mode == 'eval', is_dense=True))
            agent = RLAgent(PPO2.load('models/ppo_model.zip'))
        elif args.agent == 'script':
            print("The simulation has ended")
            env = UR10(is_train=args.mode == 'eval', is_dense=True)
            print("the simulation has started")
            agent = HardcodedAgent(UR10.position_bounds, UR10.orientation_bounds)
        elif args.agent == 'random':
            env = UR10(is_train=args.mode == 'eval', is_dense=True)
            agent = RandomAgent(env.action_space)
        else:
            assert False, f'{args.agent} is not supported'

        if args.mode in ('eval', 'viz'):
            print('success rate', evaluate(agent, env, viz=args.mode != 'eval'))
        elif args.mode == 'video':
            make_video(agent, env)
        else:
            assert False, f'{args.mode} is not supported'
