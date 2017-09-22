from gym_data import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def one_hot(action):
    n = len(set(action))
    m = len(action)

    action_onehot = np.zeros((m,n))

    for i in range(m):
        for j in range(n):
            if action[i] == j:
                action_onehot[i,j] = 1
                break

    return action_onehot


def display_state(state, gray=False):
    img = Image.fromarray(state)
    if gray:
        plt.gray()
        plt.imshow(img)
        plt.show()
    else:
        plt.imshow(img)
        plt.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def preprocess_state(state):
    img = rgb2gray(state) #grayscale
    img = (img-128)/128 -1 #restrict to [-1,1]
    return img.reshape(64,64,1)

def preprocess_all(state):
    state_pp = []
    i=0
    for s in state:
        state_pp.append(preprocess_state(s))
        print('State ', i, ' processed')
        i+=1

    return state_pp

def save_pp_dataset(state, state_prime, action, reward, file):
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('state', data=state)
        hf.create_dataset('action', data=action)
        hf.create_dataset('state_prime', data=state_prime)
        hf.create_dataset('reward', data=reward)


def load_data_single(file, name):
    with h5py.File(file, 'r') as hf:
        return hf[name][:]


def main():
    state, action, state_prime, reward = load_data(file='puck_data_v2.h5', return_reward=True)
    state_pp = preprocess_all(state)
    state_prime_pp = preprocess_all(state_prime)
    action_pp = one_hot(action)

    save_pp_dataset(state_pp, state_prime_pp, action_pp, reward, 'puck_data_v2_pp.h5')




if __name__ == '__main__':
    main()