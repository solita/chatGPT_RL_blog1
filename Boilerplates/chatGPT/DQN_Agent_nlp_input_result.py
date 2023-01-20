import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.95
        self.learning_rate = 0.001
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.batch_size:
            return
        mini_batch = random.sample(self.memory, self.batch_size)
        update_input = np.zeros((self.batch_size, self.action_size))
        update_output = np.zeros((self.batch_size, self.action_size))
        for i in range(self.batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[action] = reward + Q_future * self.discount_factor
            update_input[i] = state
            update_output[i] = target
        self.model.fit(update_input, update_output,
                       batch_size=self.batch_size, epochs=1, verbose=0)

    def save(self, name):
        self.model.save(name)
