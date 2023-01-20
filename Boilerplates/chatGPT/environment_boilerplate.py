# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [[i, j] for i in range(m) for j in range(
            m)] + [[0, 0]]  # all possible pickup and drop-off locations, plus the option to go offline
        self.state_space = [(i, j, k) for i in range(m) for j in range(
            t) for k in range(d)]  # all possible combinations of location, hour, and day
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()

    # Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros((m + t + d))
        state_encod[state[0]] = 1
        state_encod[m + state[1]] = 1
        state_encod[m + t + state[2]] = 1
        return state_encod

    # Use this function if you are using architecture-2
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
    #     return state_encod

    # Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)

        if requests > 15:
            requests = 15

        # (0,0) is not considered as customer request
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests)
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append([0, 0])

        return possible_actions_index, actions

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if action == [0, 0]:
            return -C
        else:
            pickup, dropoff = action
            time = Time_matrix[state[0], pickup] + Time_matrix[pickup, dropoff]
            reward = (time * R) - (time * C)
            return reward

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        if action == [0, 0]:
            return state
        else:
            pickup, dropoff = action
            time = Time_matrix[state[0], pickup] + Time_matrix[pickup, dropoff]
            hour = (state[1] + time) % t
            day = (state[2] + (state[1] + time) // t) % d
            return (dropoff, hour, day)

    def reset(self):
        return self.action_space, self.state_space, self.state_init
