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

    def act(self, state):
        state = state['observation']
        self.step += 1
        gripper_pos = state[:3]
        gripper_orientation = state[3:7]
        delta_pos = state[12:15]
        target_pos = np.zeros(4)
        target_orient = np.zeros(5)
        if self.step < 40:
            delta_pos[2] += 0.5
            delta = delta_pos/50 #+  np.array([0, 0, 0.5]) / self.step
            target_pos[:3] =  delta
            #target_pos[:3] = self._rescale(delta, self.position_bounds)
            target_pos[3] = -1
        elif self.step <100:
            delta_pos[2]+=0.1
            delta_pos[1]-=0.2
            # delta_pos[0] +=0.1
            delta = delta_pos/50
            target_pos[:3] = delta
            #target_pos[:3] = self._rescale(delta, self.position_bounds)
            target_pos[3] = -1
        elif self.step< 150:
            delta_pos[2]+=0.1
            delta_pos[1]-=0.1
            target_pos[:3] = delta_pos
            target_pos[3] = -1
            #target_pos= 
        else:
            delta_pos[2]+=0.1
            target_pos[:3] = delta_pos
            target_pos[3] = 10
        return target_pos, target_orient
        
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


def evaluate(policy, env, nepisodes=100, viz=False):
    success = []
    reward = []
    for episode in trange(nepisodes):
        state = env.reset()
        policy.reset()
        reward.append(0)
        while True:
            if env.step_id <= 300:
                pos, orient = policy.act(state)
                env.call_camera()
                state, rew, done, info = env.step(pos)
            else:
                pos,orient = policy.act(state)
                env.call_camera()
                state,rew,done,info = env.curl(pos)
            reward[-1] += rew
            if viz:
                time.sleep(0.01)
            if done:
                success.append(info['is_success'])
                print(success[-1])
                break
        # Calculate the angular velocity in radians per second
        throw_duration_seconds = 1  # Duration of throw in seconds
        max_angle_radians = np.random.uniform(np.pi/2,np.pi) # Maximum angle for the throw (180 degrees)
        angular_velocity_rad_per_sec = max_angle_radians / throw_duration_seconds
        print(angular_velocity_rad_per_sec)
        # Calculate the number of simulation steps
        simulation_steps = int(throw_duration_seconds / env.dt)  # Assuming env.dt is the simulation time step
        # Perform the throwing motion
        grip_val =10
        change= 0
        for _ in range(simulation_steps):
            # Convert angular velocity from rad/sec to rad/step
            env.call_camera()
            # Move the wrist joint
            if change==0:
                state, rew, done, info, change = env.throw(angular_velocity_rad_per_sec,change)
            else:
                state, rew, done, info, change = env.throw(-angular_velocity_rad_per_sec,change)
            
            if viz:
                time.sleep(0.01)

            if done:
                time.sleep(2)
                break

        # Ensure the gripper is fully opened after the throw
        # env.open_gripper()
        # state, rew, done, info = env.throw(pos)
    env.close()
    print(reward)
    return np.mean(success), np.mean(reward)


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
            env = UR10(is_train=args.mode == 'eval', is_dense=True)
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
