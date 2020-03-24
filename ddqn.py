from collections import deque
import gym
import numpy as np
import random as rand
import tensorflow as tf
import timeit
from time import time

class DdqnAgent():
    def __init__(self,
                 env='LunarLander-v2',
                 gamma=.99,
                 replay_buffer_size=50000,
                 mini_batch_size=32,
                 weight_update_interval=5000,
                 epsilon=1.0,
                 epsilon_decay=.997,
                 epsilon_min=.1,
                 learning_rate=.00025,
                 loss_function=tf.keras.losses.Huber):

        tf.random.set_seed(10)
        np.random.seed(10)
        rand.seed(10)

        self.env = gym.make(env)
        self.gamma = gamma
        self.replay_buffer_size = replay_buffer_size
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.weight_update_interval = weight_update_interval

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.actual = self.get_q_net(8, 4)  # actions
        self.target = self.get_q_net(8, 4)  # actions
        self.target.set_weights(self.actual.get_weights())
        self.loss_function = loss_function
        self._seed_replay_buffer()

    def _get_action(self, actual, state):
        # select best action, using epsilon greedy strategy
        if rand.uniform(0.0, 1.0) <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(actual.predict(np.array([state])))

    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def _seed_replay_buffer(self):
        state = self.env.reset()

        for _ in range(self.replay_buffer_size):
            action = self.env.action_space.sample()
            state_prime, reward, done, _ = self.env.step(action)

            self.replay_buffer.append(
                state, action, reward, state_prime, done)

            if done:
                state = self.env.reset()
            else:
                state = state_prime

    def _step(self, state):
        action = self._get_action(self.actual, state)

        # take step
        state_prime, reward, done, _ = self.env.step(action)

        # save observation in replay buffer
        self.replay_buffer.append(state, action, reward, state_prime, done)

        # sample mini_batch

        states, actions, state_primes, rewards, dones = self.replay_buffer.get_mini_batch(
            self.mini_batch_size)

        # calculate mini_batch q values
        actual_states = self.actual.predict(states)
        actual_s_primes = self.actual.predict(state_primes)

        target_s_primes = self.target.predict(state_primes)

        # DDQN update step
        for i in range(states.shape[0]):
                actual_states[i, actions[i]] = \
                    rewards[i] + self.gamma * target_s_primes[
                        i, np.argmax(actual_s_primes[i])] * (1 - dones[i])

        self.actual.train_on_batch(states, actual_states)

        return state_prime, reward, done

    def train(self, max_episodes):
        steps = 0
        reward_totals = deque(maxlen=100)
        for episode in range(max_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            self._update_epsilon()
            while not done:
                state_prime, reward, done = self._step(state)

                state = state_prime

                steps += 1
                total_reward += reward

                if steps % self.weight_update_interval == 0:
                    self.target.set_weights(self.actual.get_weights())

            reward_totals.append(total_reward)
            if total_reward > 160:
                self.actual.optimizer.lr.assign(.0001)
                self.target.optimizer.lr.assign(.0001)
            if len(reward_totals) == 100:
                print("episode: {0}, avg_reward: {1}".format(
                    episode, np.sum(reward_totals) / 100))
                # check if won or time to quit
                if sum(reward_totals) / 100 >= 200.0 or episode == max_episodes:
                    test_avg = test(self.actual, self.env)
                    if test_avg > 200.0 or episode == max_episodes:
                        if test_avg > 200:
                            print('win!')
                            print(test_avg)
                        break

    def get_q_net(self, input_dim, output_dim):
        model = tf.keras.Sequential([
            # Adds a densely-connected layer with 64 units to the model:
            tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            # Add another:
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='linear')])

        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.learning_rate),
                      loss=tf.keras.losses.Huber())

        return model

    # loss function implementation taken from:
    # https://github.com/dennisfrancis/LunarLander-v2/blob/master/src/run_full_dqn.py


class ReplayBuffer():
    def __init__(self, size):
        self.state_buffer = np.zeros((size, 8))
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.reward_buffer = np.zeros(size)
        self.state_prime_buffer = np.zeros((size, 8))
        self.done_buffer = np.zeros(size)

        self.insertion_index = 0

    def append(self, state, action, reward, state_prime, done):
        self.state_buffer[self.insertion_index] = state
        self.action_buffer[self.insertion_index] = action
        self.state_prime_buffer[self.insertion_index] = state_prime
        self.reward_buffer[self.insertion_index] = reward
        self.done_buffer[self.insertion_index] = done

        if self.insertion_index == self.state_buffer.shape[0] - 1:
            self.insertion_index = 0
        else:
            self.insertion_index += 1

    def get_mini_batch(self, mini_batch_size):        
        # probabilities = (((self.reward_buffer - self.reward_buffer.min()) / (self.reward_buffer.max() - self.reward_buffer.min()))/(self.reward_buffer.size/2))
        probabilities = np.random.random(self.state_buffer.shape[0])
        probabilities /= np.sum(probabilities)
        random_shifts = np.random.random(probabilities.shape)
        random_shifts /= random_shifts.sum()
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - probabilities
        indices =  np.argpartition(shifted_probabilities, mini_batch_size)[:mini_batch_size]
        return (self.state_buffer[indices],
                self.action_buffer[indices],
                self.state_prime_buffer[indices],
                self.reward_buffer[indices],
                self.done_buffer[indices])


def test(actual, env):
    total_reward = 0
    episode_rewards = []
    for i in range(0, 100):
        done = False
        s = env.reset()
        episode_r = 0

        while not done:
            a = np.argmax(actual.predict(np.array([s])))
            s_prime, r, done, _ = env.step(a)
            s = s_prime
            episode_r += r

        episode_rewards.append(episode_r)
        total_reward += episode_r

    return total_reward / 100


if __name__ == "__main__":
    agent = DdqnAgent()
    agent.train(5000)
    