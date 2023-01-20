import random
from collections import deque

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Hyperparameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.0001
        self.epsilon_min = 0.01

        self.batch_size = 32
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()

    def build_model(self):
        """Build the neural network model for the DQN"""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.summary
        return model

    def get_action(self, state):
        """get action from model using epsilon-greedy policy"""
        epsilon = self.epsilon_min + \
            (self.epsilon_max - self.epsilon_min) * \
            np.exp(-self.epsilon_decay*episode)
        if np.random.rand() <= epsilon:
            # explore: choose a random action
            return random.randrange(self.action_size)
        else:
            # exploit: choose the action with the highest predicted Q-value
            return np.argmax(self.model.predict(state))

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            target = reward
            if not done:
                target = reward + self.discount_factor * \
                    np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

    def save(self, name):
        self.model.save(name)
