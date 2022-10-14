
import acme
import dm_env
from dm_env._environment import TimeStep
from dm_env import specs
import numpy as np


class CoinFlip(dm_env.Environment):

    def __init__(self):
        self.funds = 50
        self.time=0
        # self._observation_spec = specs.DiscreteArray(
        #     dtype=int, num_values=10, name="observation")
        # self._action_spec = specs.DiscreteArray(
        #     dtype=int, num_values=10, name="action")
        self._reset_next_step = True

    def step(self, action) -> TimeStep:
        if self._reset_next_step:
            return self.reset()
        self.time+=1
        flip = np.random.choice(["heads", "tails"], )

        print(f'State: {self.time} Bet: {action} Coin flip: {flip} Funds: {self.funds}')
        if flip == "tails":
            action = action*-1
        self.funds+=action
        if self.funds==0:
            self._reset_next_step = True
            return dm_env.termination(reward=self.funds, observation=self._observation())
        elif self.time==9:
            self._reset_next_step = True
            return dm_env.termination(reward=self.funds-51, observation=self._observation())
        else:
            return dm_env.transition(reward=action, observation=self._observation(), discount=1/(self._observation()[0]+1))


    def observation_spec(self):
        return specs.Array((2,),
            dtype=int, name="observation")

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=10, name="action")

    def reset(self) -> TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self.time = 0
        self.funds = 50
        observation = self._observation()
        return dm_env.restart(observation)

    def _observation(self):
        return self.time,self.funds,