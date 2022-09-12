import acme
import numpy as np


# epsilon greedy policy
def epsilon_greedy(q_values, epsilon):
    if epsilon < np.random.random():
        return np.argmax(q_values)
    else:
        return np.random.choice(len(q_values))

class QLearningAgent(acme.Actor):

    def __init__(self, env_specs=None, step_size=0.1):
        self.Q = np.random.random((10,10))

        # set step size
        self.step_size = step_size

        # set behavior policy
        # self.policy = None
        # self.behavior_policy = lambda q_values: epsilon_greedy(q_values, epsilon=0.1)
        self.behavior_policy = lambda q_values: np.argmax(q_values)

        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None

    def state_to_index(self, state):
        state = *map(int, state),
        return state

    def transform_state(self, state):
        # this is specifally required for the blackjack environment
        # state = *map(int, state[0]),
        # state[0][2] = int(state[0][2])
        return state

    def select_action(self, observation):
        state = self.transform_state(observation)
        a = self.behavior_policy(self.Q[state])
        print(a)
        return a

    def observe_first(self, timestep):
        self.timestep = timestep

    def observe(self, action, next_timestep):
        self.action = action
        self.next_timestep = next_timestep

    def update(self):
        # get variables for convenience
        state = self.timestep.observation
        _, reward, discount, next_state = self.next_timestep
        action = self.action

        # turn states into indices
        state = self.transform_state(state)
        next_state = self.transform_state(next_state)

        # Q-value update
        td_error = reward + discount * np.max(self.Q[next_state]) - self.Q[state][action]
        self.Q[state][action] += self.step_size * td_error

        # finally, set timestep to next_timestep
        self.timestep = self.next_timestep