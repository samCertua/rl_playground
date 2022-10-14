import random

class RandomAgent:

    def __init__(self, action_space, observation_space):
        self.action_space = action_space


    def select_action(self, observation):
        return random.randint(0,self.action_space)

    def observe_first(self, observation):
        pass
    def observe(self, action, reward, observation, last):
        pass

    def update(self, deployment=False):
        pass