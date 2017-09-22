import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from math import pi

class TransitionModelV2():
    def __init__(self):
        self.save_path = 'checkpoints/reward_model_v3'

        self.from_scratch = False
        self.pixel_dim = [64,64,1]
        self.state_size = 64*64 #784
        self.action_size = 5 #one-hot encoder of action choice
        self.n_z = 30
        self.batchsize = 64
        self.r_estimates = 10
        self.epochs = 1
        self.BETA = 1
        self.LAMBDA = 1
        self.learning_rate = .00001

        self.state = tf.placeholder(tf.float32, [None, self.pixel_dim[0], self.pixel_dim[1], self.pixel_dim[2]])
        self.state_prime = tf.placeholder(tf.float32, [None, self.pixel_dim[0], self.pixel_dim[1], self.pixel_dim[2]])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.reward = tf.placeholder(tf.float32)
        self.sample = tf.placeholder(tf.float32, [None, self.n_z])

        self.samples_1 = tf.random_normal([self.batchsize, self.n_z],0,1,dtype=tf.float32)
        self.samples_2 = tf.random_normal([self.batchsize, self.n_z],0,1,dtype=tf.float32)

        self.mu_s, self.sigma_s = self.stateEncoder(self.state, reuse=None, scope='encoder')
        self.guessed_z = self.mu_s + (self.sigma_s * self.samples_1)
        self.mu_z_prime, self.sigma_z_prime = self.transition(self.guessed_z, self.action, reuse=None)
        self.guessed_z_prime = self.mu_z_prime + (self.sigma_z_prime * self.samples_2)
        self.mu_s_prime, self.sigma_s_prime = self.stateEncoder(self.state_prime, reuse=True, scope='encoder')
        self.mu_r, self.sigma_r= self.rewardGenerator(self.guessed_z_prime, reuse=None)

        self.mu, self.sigma = self.stateEncoder(self.state, reuse=True, scope='encoder') #these are for later use when predicting
        self.mu_t, self.sigma_t = self.transition(self.sample, self.action, reuse = True)
        self.mu_re, self.sigma_re = self.rewardGenerator(self.sample, reuse=True)

        self.transition_loss = 0.5 * tf.reduce_sum(
            tf.log(tf.square(self.sigma_s_prime)+1e-8) - tf.log(tf.square(self.sigma_z_prime)+1e-8) + tf.realdiv(tf.square(self.sigma_z_prime), tf.square(self.sigma_s_prime)+1e-8)
            + tf.realdiv(tf.square(self.mu_z_prime - self.mu_s_prime), tf.square(self.sigma_s_prime)+1e-8) - 1, 1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mu_s) + tf.square(self.sigma_s) - tf.log(1e-8+tf.square(self.sigma_s)) - 1,1)
        #self.reward_loss = self.reward_loss_func(self.samples_2, self.reward, self.mu_z_prime, self.sigma_z_prime)
        self.reward_loss = -0.5*tf.log(self.sigma_r) - tf.square(tf.realdiv(self.reward - self.mu_r, 2*self.sigma_r))
        self.cost = tf.reduce_mean(self.LAMBDA*self.transition_loss + self.BETA*self.latent_loss + self.reward_loss)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

    ### H(r|z) = E[-log(p(r|z))] ~= (1/N)sum_i(-log(p(r_j | z_j,i))) = (1/N)sum_i [(log(1/(sqrt(2pi)*sigma)) - (r-mu)^2/2*sigma^2]
    def reward_loss_func(self, samples, reward, mu_z_prime, sigma_z_prime):
        loss = 0
        for i in range(self.r_estimates):
            guess = mu_z_prime + (sigma_z_prime*samples[i,:,:])
            if i==0:
                mu, sigma = self.rewardGenerator(guess, reuse=None)
            else:
                mu, sigma = self.rewardGenerator(guess, reuse=True)
            loss += -0.5*tf.log(sigma) - tf.square(tf.realdiv(reward - self.mu, 2*sigma))

        loss /= self.r_estimates

        return loss

    def transition_loss_func(self, samples, mu_s_prime, sigma_s_prime):
        pass

    def stateEncoder(self, state, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):

            state = tf.reshape(state, [-1, 64,64,1])

            ##size: [batchsize, 64, 64, 1]
            conv1 = tf.layers.conv2d(inputs=state, filters = 32, kernel_size = 5, padding = "same", activation = tf.nn.relu)
            conv1 = tf.layers.batch_normalization(inputs=conv1)
            ##size: [batchsize, 64, 64, 32]
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 32, 32, 32]
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 32, kernel_size = 5, padding= "same", activation = tf.nn.relu)
            conv2 = tf.layers.batch_normalization(inputs=conv2)
            ##size: [batchsize, 32, 32, 32]
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 16, 16, 32]
            conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(inputs=conv3)
            ##size: [batchsize, 32, 32, 32]
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 8, 8, 32]

            pool2_flat = tf.reshape(pool3, [-1, 8*8*32])
            dense = tf.layers.dense(inputs=pool2_flat, units=8*8*32, activation=tf.nn.relu)

            w_mean = tf.layers.dense(inputs=dense, units=self.n_z, name="w_mean", activation=tf.nn.tanh)
            w_stddev = tf.layers.dense(inputs=dense, units=self.n_z, name="w_stddev")
            w_stddev = tf.nn.softplus(w_stddev)

        return w_mean, w_stddev

    def transition(self, z, a, reuse):
        with tf.variable_scope('transition', reuse=reuse):
            input_layer = tf.concat([z, a], 1)
            dense1 = tf.layers.dense(inputs=input_layer, units=self.n_z, activation=tf.nn.relu)
            dense1 = tf.layers.batch_normalization(inputs=dense1)

            dense2 = tf.layers.dense(inputs=dense1, units=self.n_z, activation=tf.nn.relu)
            dense2 = tf.layers.batch_normalization(inputs=dense2)

            z_mean = tf.layers.dense(inputs=dense2, units= self.n_z, name="z_mean", activation=tf.nn.tanh)
            z_stddev = tf.layers.dense(inputs=dense2, units= self.n_z, name="z_stddev")
            z_stddev = tf.nn.softplus(z_stddev)

        return z_mean, z_stddev

    def rewardGenerator(self, z, reuse):
        with tf.variable_scope('reward_generator', reuse=reuse):
            dense1 = tf.layers.dense(inputs=z, units=self.n_z, activation=tf.nn.relu)
            dense1 = tf.layers.batch_normalization(inputs=dense1)

            dense2 = tf.layers.dense(inputs=dense1, units=int(self.n_z/2), activation=tf.nn.relu)
            dense2 = tf.layers.batch_normalization(inputs=dense2)

            r_mean = tf.layers.dense(inputs=dense2, units= 1, name="r_mean", activation=tf.nn.sigmoid)
            r_stddev = tf.layers.dense(inputs=dense2, units= 1, name="r_stddev")
            r_stddev = tf.nn.softplus(r_stddev)

        return r_mean, r_stddev


    def enc_func(self, state):
        state = np.reshape(state, (-1,64,64,1))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.save_path)
            mu, sigma = sess.run([self.mu, self.sigma], feed_dict={self.state: state})

        return mu, sigma

    def trans_func(self, sample, action):
        sample = np.reshape(sample, (-1,self.n_z))
        action = np.reshape(action, (-1,self.action_size))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.save_path)
            mu, sigma = sess.run([self.mu_t, self.sigma_t], feed_dict={self.sample: sample, self.action: action })

        return mu, sigma

    def reward_func(self, sample):
        sample = np.reshape(sample, (-1,self.n_z))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)
            mu, sigma = sess.run([self.mu_re, self.sigma_re], feed_dict={self.sample: sample})

        return mu, sigma

    def train(self, state_data, action_data, state_prime_data, reward_data):
        N = len(state_data)
        best_loss = 1e8
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if not self.from_scratch:
                saver.restore(sess, save_path=self.save_path)

            for epoch in range(self.epochs):
                training_batch = zip(range(0, N, self.batchsize),
                                     range(self.batchsize, N+1, self.batchsize))

                state_data, action_data, state_prime_data, reward_data = shuffle(state_data, action_data, state_prime_data, reward_data)

                print('Epoch: ', epoch)
                batch_best_loss = best_loss
                iter = 0
                for start, end in training_batch:
                    _, loss_val = sess.run([self.optimizer,self.cost], feed_dict={self.state: state_data[start:end], self.action: action_data[start:end], self.state_prime: state_prime_data[start:end], self.reward: reward_data[start:end]})

                    if loss_val < batch_best_loss:
                        batch_best_loss = loss_val

                    if iter%10 == 0:
                        print("Iter: ", iter, "Loss: ", loss_val)

                    iter += 1
                if batch_best_loss < best_loss:
                    best_loss = batch_best_loss
                    saver.save(sess, save_path=self.save_path)


    def class_test(self):
        print('Class loaded')

class TransitionModel():
    def __init__(self):
        self.save_path = 'checkpoints/reward_model_v2'

        self.from_scratch = False
        self.pixel_dim = [64,64,1]
        self.state_size = 64*64 #784
        self.action_size = 5 #one-hot encoder of action choice
        self.n_z = 30
        self.batchsize = 64
        self.r_estimates = 50
        self.epochs = 1
        self.BETA = 1
        self.LAMBDA = 1
        self.learning_rate = .00001

        self.state = tf.placeholder(tf.float32, [None, self.pixel_dim[0], self.pixel_dim[1], self.pixel_dim[2]])
        self.state_prime = tf.placeholder(tf.float32, [None, self.pixel_dim[0], self.pixel_dim[1], self.pixel_dim[2]])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.reward = tf.placeholder(tf.float32)
        self.sample = tf.placeholder(tf.float32, [None, self.n_z])

        self.samples_1 = tf.random_normal([self.batchsize, self.n_z],0,1,dtype=tf.float32)
        self.samples_2 = tf.random_normal([self.batchsize, self.n_z],0,1,dtype=tf.float32)

        self.mu_s, self.sigma_s = self.stateEncoder(self.state, reuse=None, scope='encoder')
        #self.guessed_z = self.mu_s + (self.sigma_s * self.samples_1)
        self.mu_z_prime, self.sigma_z_prime = self.transition(self.mu_s, self.action, reuse=None)
        #self.guessed_z_prime = self.mu_z_prime + (self.sigma_z_prime * self.samples_2)
        self.mu_s_prime, self.sigma_s_prime = self.stateEncoder(self.state_prime, reuse=True, scope='encoder')
        self.mu_r, self.sigma_r = self.rewardGenerator(self.mu_s_prime, reuse=None)

        self.mu, self.sigma = self.stateEncoder(self.state, reuse=True, scope='encoder') #these are for later use when predicting
        self.mu_t, self.sigma_t = self.transition(self.sample, self.action, reuse = True)
        self.mu_re, self.sigma_re = self.rewardGenerator(self.sample, reuse=True)

        self.transition_loss = 0.5 * tf.reduce_sum(
            tf.log(tf.square(self.sigma_s_prime)+1e-8) - tf.log(tf.square(self.sigma_z_prime)+1e-8) + tf.realdiv(tf.square(self.sigma_z_prime), tf.square(self.sigma_s_prime)+1e-8)
            + tf.realdiv(tf.square(self.mu_z_prime - self.mu_s_prime), tf.square(self.sigma_s_prime)+1e-8) - 1, 1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.mu_s) + tf.square(self.sigma_s) - tf.log(1e-8+tf.square(self.sigma_s)) - 1,1)
        #self.reward_loss = self.reward_loss_func(self.samples_2, self.reward, self.mu_z_prime, self.sigma_z_prime)
        self.reward_loss = -0.5*tf.log(self.sigma_r) - tf.square(tf.realdiv(self.reward - self.mu_r, 2*self.sigma_r))
        self.cost = tf.reduce_mean(self.LAMBDA*self.transition_loss + self.BETA*self.latent_loss - self.reward_loss)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

    ### H(r|z) = E[-log(p(r|z))] ~= (1/N)sum_i(-log(p(r_j | z_j,i))) = (1/N)sum_i [(log(1/(sqrt(2pi)*sigma)) - (r-mu)^2/2*sigma^2]
    #def reward_loss_func(self, samples, reward, mu_z_prime, sigma_z_prime):
    #    loss = 0
    #    for i in range(self.r_estimates):
    #        guess = mu_z_prime + (sigma_z_prime*samples[i,:,:])
    #        if i==0:
    #            mu, sigma = self.rewardGenerator(guess, reuse=None)
    #        else:
    #            mu, sigma = self.rewardGenerator(guess, reuse=True)
    #        loss += tf.log(1e-8 + (1/(tf.sqrt(2*pi)*sigma))) - tf.realdiv(tf.square(reward - mu), 2*tf.square(sigma))

    #    loss /= self.r_estimates

    #    return loss


    def stateEncoder(self, state, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):

            state = tf.reshape(state, [-1, 64,64,1])

            ##size: [batchsize, 64, 64, 1]
            conv1 = tf.layers.conv2d(inputs=state, filters = 32, kernel_size = 5, padding = "same", activation = tf.nn.relu)
            conv1 = tf.layers.batch_normalization(inputs=conv1)
            ##size: [batchsize, 64, 64, 32]
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 32, 32, 32]
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 64, kernel_size = 5, padding= "same", activation = tf.nn.relu)
            conv2 = tf.layers.batch_normalization(inputs=conv2)
            ##size: [batchsize, 32, 32, 64]
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 16, 16, 64]
            conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(inputs=conv3)
            ##size: [batchsize, 32, 32, 64]
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 8, 8, 32]

            pool2_flat = tf.reshape(pool3, [-1, 8*8*32])
            dense = tf.layers.dense(inputs=pool2_flat, units=8*8*32, activation=tf.nn.relu)

            w_mean = tf.layers.dense(inputs=dense, units=self.n_z, name="w_mean", activation=tf.nn.tanh)
            w_stddev = tf.layers.dense(inputs=dense, units=self.n_z, name="w_stddev")
            w_stddev = tf.nn.softplus(w_stddev)

        return w_mean, w_stddev

    def transition(self, z, a, reuse):
        with tf.variable_scope('transition', reuse=reuse):
            input_layer = tf.concat([z, a], 1)
            dense1 = tf.layers.dense(inputs=input_layer, units=self.n_z, activation=tf.nn.relu)
            dense1 = tf.layers.batch_normalization(inputs=dense1)

            dense2 = tf.layers.dense(inputs=dense1, units=self.n_z, activation=tf.nn.relu)
            dense2 = tf.layers.batch_normalization(inputs=dense2)

            z_mean = tf.layers.dense(inputs=dense2, units= self.n_z, name="z_mean", activation=tf.nn.tanh)
            z_stddev = tf.layers.dense(inputs=dense2, units= self.n_z, name="z_stddev")
            z_stddev = tf.nn.softplus(z_stddev)

        return z_mean, z_stddev

    def rewardGenerator(self, z, reuse):
        with tf.variable_scope('reward_generator', reuse=reuse):
            dense1 = tf.layers.dense(inputs=z, units=self.n_z, activation=tf.nn.relu)
            dense1 = tf.layers.batch_normalization(inputs=dense1)

            dense2 = tf.layers.dense(inputs=dense1, units=int(self.n_z/2), activation=tf.nn.relu)
            dense2 = tf.layers.batch_normalization(inputs=dense2)

            r_mean = tf.layers.dense(inputs=dense2, units= 1, name="r_mean", activation=tf.nn.sigmoid)
            r_stddev = tf.layers.dense(inputs=dense2, units= 1, name="r_stddev")
            r_stddev = tf.nn.softplus(r_stddev)

        return r_mean, r_stddev


    def enc_func(self, state):
        state = np.reshape(state, (-1,64,64,1))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.save_path)
            mu, sigma = sess.run([self.mu, self.sigma], feed_dict={self.state: state})

        return mu, sigma

    def trans_func(self, sample, action):
        sample = np.reshape(sample, (-1,self.n_z))
        action = np.reshape(action, (-1,self.action_size))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.save_path)
            mu, sigma = sess.run([self.mu_t, self.sigma_t], feed_dict={self.sample: sample, self.action: action })

        return mu, sigma

    def reward_func(self, sample):
        sample = np.reshape(sample, (-1,self.n_z))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path=self.save_path)
            mu, sigma = sess.run([self.mu_re, self.sigma_re], feed_dict={self.sample: sample})

        return mu, sigma

    def train(self, state_data, action_data, state_prime_data, reward_data):
        N = len(state_data)
        best_loss = 1e8
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if not self.from_scratch:
                saver.restore(sess, save_path=self.save_path)

            for epoch in range(self.epochs):
                training_batch = zip(range(0, N, self.batchsize),
                                     range(self.batchsize, N+1, self.batchsize))

                state_data, action_data, state_prime_data, reward_data = shuffle(state_data, action_data, state_prime_data, reward_data)

                print('Epoch: ', epoch)
                batch_best_loss = best_loss
                iter = 0
                for start, end in training_batch:
                    _, loss_val = sess.run([self.optimizer,self.cost], feed_dict={self.state: state_data[start:end], self.action: action_data[start:end], self.state_prime: state_prime_data[start:end], self.reward: reward_data[start:end]})

                    if loss_val < batch_best_loss:
                        batch_best_loss = loss_val

                    if iter%10 == 0:
                        print("Iter: ", iter, "Loss: ", loss_val)

                    iter += 1
                if batch_best_loss < best_loss:
                    best_loss = batch_best_loss
                    saver.save(sess, save_path=self.save_path)


    def class_test(self):
        print('Class loaded')


class TransitionModelNoRewards():
    def __init__(self):
        self.save_path = 'checkpoints/batchnorm_weightshare'

        self.from_scratch = False
        self.pixel_dim = [64,64,1]
        self.state_size = 64*64 #784
        self.action_size = 5 #one-hot encoder of action choice
        self.n_z = 15
        self.batchsize = 64
        self.r_estimates = 50
        self.epochs = 1
        self.BETA = 1
        self.learning_rate = .00001

        self.state = tf.placeholder(tf.float32, [None, self.pixel_dim[0], self.pixel_dim[1], self.pixel_dim[2]])
        self.state_prime = tf.placeholder(tf.float32, [None, self.pixel_dim[0], self.pixel_dim[1], self.pixel_dim[2]])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.reward = tf.placeholder(tf.float32)
        self.sample = tf.placeholder(tf.float32, [None, self.n_z])

        samples_1 = tf.random_normal([self.batchsize, self.n_z],0,1,dtype=tf.float32)
        samples_2 = tf.random_normal([self.batchsize, self.n_z],0,1)

        mu_z1, sigma_z1 = self.stateEncoder(self.state, reuse=None, name='encoder')
        guessed_z = mu_z1 + (sigma_z1 * samples_1)
        mu_z2, sigma_z2 = self.transition(guessed_z, self.action, reuse=None)
        guessed_z_prime = mu_z2 + (sigma_z2*samples_2)
        mu_s, sigma_s = self.stateEncoder(self.state_prime, reuse=True, name='encoder')

        self.mu, self.sigma = self.stateEncoder(self.state, reuse=True, name='encoder')
        self.mu_t, self.sigma_t = self.transition(self.sample, self.action, reuse = True)

        self.transition_loss = 0.5 * tf.reduce_sum(
            tf.log(tf.square(sigma_s)+1e-8) - tf.log(tf.square(sigma_z2)+1e-8) + tf.realdiv(tf.square(sigma_z2), tf.square(sigma_s)+1e-6)
            + tf.realdiv(tf.square(mu_z2 - mu_s), tf.square(sigma_s)+1e-6) - 1, 1)

        #self.transition_loss = self.transition_loss_func(samples_2, mu_s, mu_z2, sigma_z2)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(mu_z1) + tf.square(sigma_z1) - tf.log(1e-8+tf.square(sigma_z1)) - 1,1)
        #self.latent_loss2 = 0.5 * tf.reduce_sum(tf.square(mu_s) + tf.square(sigma_s) - tf.log(1e-8+tf.square(sigma_s)) - 1,1)

        #self.reward_loss = tf.reduce_sum(tf.square(self.reward_pred - self.reward))
        self.cost = tf.reduce_mean(self.transition_loss + self.BETA*self.latent_loss)#  + self.reward_loss) # + self.latent_loss2)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

    def stateEncoder(self, state, reuse, name):
        with tf.variable_scope(name, reuse=reuse):

            state = tf.reshape(state, [-1, 64,64,1])

            ##size: [batchsize, 64, 64, 1]
            conv1 = tf.layers.conv2d(inputs=state, filters = 32, kernel_size = 5, padding = "same", activation = tf.nn.relu)
            conv1 = tf.layers.batch_normalization(inputs=conv1)
            ##size: [batchsize, 64, 64, 32]
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 32, 32, 32]
            conv2 = tf.layers.conv2d(inputs=pool1, filters = 64, kernel_size = 5, padding= "same", activation = tf.nn.relu)
            conv2 = tf.layers.batch_normalization(inputs=conv2)
            ##size: [batchsize, 32, 32, 64]
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 16, 16, 64]
            conv3 = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(inputs=conv3)
            ##size: [batchsize, 32, 32, 64]
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            ##size: [batchsize, 8, 8, 32]

            pool2_flat = tf.reshape(pool3, [-1, 8*8*32])
            dense = tf.layers.dense(inputs=pool2_flat, units=8*8*32, activation=tf.nn.relu)

            w_mean = tf.layers.dense(inputs=dense, units=self.n_z, name="w_mean", activation=tf.nn.tanh)
            w_stddev = tf.layers.dense(inputs=dense, units=self.n_z, name="w_stddev")
            w_stddev = tf.nn.softplus(w_stddev)

        return w_mean, w_stddev

    def transition(self, z, a, reuse):
        with tf.variable_scope('transition', reuse=reuse):
            input_layer = tf.concat([z, a], 1)
            dense1 = tf.layers.dense(inputs=input_layer, units=self.n_z, activation=tf.nn.relu)
            dense1 = tf.layers.batch_normalization(inputs=dense1)

            dense2 = tf.layers.dense(inputs=dense1, units=self.n_z, activation=tf.nn.relu)
            dense2 = tf.layers.batch_normalization(inputs=dense2)

            z_mean = tf.layers.dense(inputs=dense2, units= self.n_z, name="z_mean", activation=tf.nn.tanh)
            z_stddev = tf.layers.dense(inputs=dense2, units= self.n_z, name="z_stddev")
            z_stddev = tf.nn.softplus(z_stddev)

        return z_mean, z_stddev

    def enc_func(self, state):
        state = np.reshape(state, (-1,64,64,1))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.save_path)
            mu, sigma = sess.run([self.mu, self.sigma], feed_dict={self.state: state})

        return mu, sigma

    def trans_func(self, sample, action):
        sample = np.reshape(sample, (-1,self.n_z))
        action = np.reshape(action, (-1,self.action_size))

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, save_path = self.save_path)
            mu, sigma = sess.run([self.mu_t, self.sigma_t], feed_dict={self.sample: sample, self.action: action })

        return mu, sigma

    def train(self, state_data, action_data, state_prime_data):
        N = len(state_data)
        best_loss = 100000
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            if not self.from_scratch:
                saver.restore(sess, save_path=self.save_path)

            for epoch in range(self.epochs):
                training_batch = zip(range(0, N, self.batchsize),
                                     range(self.batchsize, N+1, self.batchsize))

                state_data, action_data, state_prime_data = shuffle(state_data, action_data, state_prime_data)

                print('Epoch: ', epoch)
                batch_best_loss = best_loss
                iter = 0
                for start, end in training_batch:
                    _, loss_val = sess.run([self.optimizer,self.cost], feed_dict={self.state: state_data[start:end], self.action: action_data[start:end], self.state_prime: state_prime_data[start:end]})

                    if loss_val < batch_best_loss:
                        batch_best_loss = loss_val

                    if iter%10 == 0:
                        print("Iter: ", iter, "Loss: ", loss_val)

                    iter += 1
                if batch_best_loss < best_loss:
                    best_loss = batch_best_loss
                    saver.save(sess, save_path=self.save_path)


    def class_test(self):
        print('Class loaded')