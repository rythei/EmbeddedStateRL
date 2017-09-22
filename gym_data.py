import gym_ple as gym
import h5py
from pathlib import Path
#import numpy as np



def load_data(return_reward = False, file = 'data.h5'):

    my_file = Path(file)

    if my_file.is_file():
        with h5py.File(file, 'r') as hf:
            if return_reward:
                return hf['state'][:], hf['action'][:], hf['state_prime'][:], hf['reward'][:]
            else:
                return hf['state'][:], hf['action'][:], hf['state_prime'][:]
    else:
        print('No data saved, sampling new data')
        with h5py.File(file, 'w') as hf:
            state, action, state_prime, reward = sample_data(return_reward=True)
            hf.create_dataset('state', data=state)
            hf.create_dataset('action', data=action)
            hf.create_dataset('state_prime', data=state_prime)
            hf.create_dataset('reward', data=reward)

        with h5py.File(file, 'r') as hf:
            if return_reward:
                return hf['state'][:], hf['action'][:], hf['state_prime'][:], hf['reward'][:]
            else:
                return hf['state'][:], hf['action'][:], hf['state_prime'][:]


def sample_data(iters = 100, max_len = 1000, return_reward = False):

    env = gym.make('PuckWorld-v0')
    state_primes = []
    states = []
    actions = []
    rewards = []

    episodes = iters*max_len
    for i_episode in range(iters):
        observation = env.reset()
        print('EPISODE: ', i_episode)
        for t in range(max_len):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #print(reward)

            if t<max_len-1:
                states.append(observation)
                actions.append(action)

            if t > 0:
                state_primes.append(observation) #next observation and reward
                rewards.append(reward)

            #if t==max_len-1:
            #    state_primes.append(observation)

            print('ITER: ', t)

            if done:
                break

    if return_reward:
        return states, actions, state_primes, rewards
    else:
        return states, actions, state_primes


if __name__ == '__main__':
    state, action, state_prime, reward = load_data(file='puck_data_v2.h5', return_reward=True)