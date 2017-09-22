#### current model stored at checkpoint dir ####
#### checkpoints/reward_model_v3 ####

from models import TransitionModelV2
import h5py
import numpy as np

TRAIN = True

def load_data(file):
    with h5py.File(file,'r') as hf:
        return hf['state'][:], hf['action'][:], hf['state_prime'][:], hf['reward'][:]

def main():
    model = TransitionModelV2()

    model.from_scratch = False #set to True if you want to start the training from scratch, otherwise will load latest model
                               #note: if you change the above to True, you should also change the checkpoints directory
                               #to make sure you don't overwrite the current model
    model.epochs = 1000
    model.learning_rate = .00001

    state, action, state_prime, reward = load_data('puck_data_v2_pp.h5')

    rmin = np.min(reward)
    rmax = np.max(reward)

    #restrict the range of the reward to [0,1] -- makes the regression problem (p(r|z')) easier
    reward_pp = (reward-rmin)/(rmax-rmin)

    ### train ###

    if TRAIN:
        model.train(state, action, state_prime,reward_pp)

    ### check an example transition ###

    if not TRAIN:

        mu1, sigma1 = model.enc_func(state[107])
        sample = np.random.normal(0, 1, model.n_z)
        guess = mu1 + sigma1 * sample

        print('Sample from p(z|s): ', mu1)
        mu2, sigma2 = model.trans_func(guess, action[107])

        sample2 = np.random.normal(0, 1, model.n_z)
        guess2 = mu2 + sigma2 * sample2

        print('Sample from p(z_prime|z,a): ', mu2)

        mu4, sigma4 = model.enc_func(state_prime[107])

        print('Sample from p(z_prime|s_prime): ', mu4)

        mu3, sigma3 = model.reward_func(guess2)
        r_out = mu3 * (rmax - rmin) + rmin

        print('Predicted Reward: ', r_out)
        print('Actual Reward: ', reward[107])



if __name__ == '__main__':
    main()