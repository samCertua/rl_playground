import random

import acme
import dm_env
from dm_env._environment import TimeStep
from dm_env import specs


class TestEnvAddition(dm_env.Environment):

    def __init__(self):
        self.time = 0
        self.magic_number = 1
        # self._observation_spec = specs.DiscreteArray(
        #     dtype=int, num_values=10, name="observation")
        # self._action_spec = specs.DiscreteArray(
        #     dtype=int, num_values=10, name="action")
        self._reset_next_step = True

    def step(self, action) -> TimeStep:
        if self._reset_next_step:
            return self.reset()
        self.time+=1
        # self.magic_number = random.randint(0,10)
        self.magic_number = (self.magic_number + 1) % 4
        print(action)
        if action == self.magic_number:
            reward = 1
        else:
            reward = -1
        if self.time == 5:
            self._reset_next_step = True
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=reward, observation=self._observation(), discount=1 / (self.time + 1))


    def observation_spec(self):
        return specs.Array((2,),
                           dtype=int, name="observation")

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=5, name="action")

    def reset(self) -> TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self.time = 0
        self.magic_number = 4
        observation = self._observation()
        return dm_env.restart(observation)

    def _observation(self):
        return self.time, self.magic_number