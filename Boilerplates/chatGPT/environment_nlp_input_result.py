import numpy as np


class CabDriver:
    def __init__(self, num_cities, num_hours, num_days):
        self.action_space = [(i, j) for i in range(num_cities)
                             for j in range(num_cities) if i != j] + [(0, 0)]
        self.action_space = np.array(self.action_space)
        self.state_space = [(i, j, k) for i in range(num_cities)
                            for j in range(num_hours) for k in range(num_days)]
        self.state_init = np.random.choice(self.state_space)
        self.num_cities = num_cities
        self.num_hours = num_hours
        self.num_days = num_days

    def convert_state_to_vector(self, state):
        location, hour, day = state
        vec = np.zeros(self.num_cities + self.num_hours + self.num_days)
        vec[location] = 1
        vec[self.num_cities + hour] = 1
        vec[self.num_cities + self.num_hours + day] = 1
        return vec

    def replay(self, state, avg_rate):
        location, hour, day = state
        requests = np.random.poisson(avg_rate[location])
        actions = np.random.choice(self.action_space, requests, replace=True)
        if len(actions) == 0:
            actions = [(0, 0)]
        return actions

    def get_reward(self, state, action, time_matrix, fuel_cost):
        location, hour, day = state
        pickup, dropoff = action
        if pickup == 0 and dropoff == 0:
            return 0
        else:
            travel_time = time_matrix[location][pickup] + \
                time_matrix[pickup][dropoff]
            arrive_time = (hour + travel_time) % 24
            arrive_day = (day + (hour + travel_time) // 24) % 7
            revenue = (dropoff - pickup) * 2
            cost = fuel_cost * travel_time
            return revenue - cost

    def get_next_state(self, state, action, time_matrix):
        location, hour, day = state
        pickup, dropoff = action
        if pickup == 0 and dropoff == 0:
            return state
        else:
            travel_time = time_matrix[location][pickup] + \
                time_matrix[pickup][dropoff]
            arrive_time = (hour + travel_time) % 24
            arrive_day = (day + (hour + travel_time) // 24) % 7
            return (dropoff, arrive_time, arrive_day)

    def reset(self):
        self.state_init = np.random.choice(self.state_space)
        return self.action_space, self.state_space, self.state_init
