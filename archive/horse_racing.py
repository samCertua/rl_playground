
import acme
import dm_env
from dm_env._environment import TimeStep
from dm_env import specs
import numpy as np


class HorseRacing(dm_env.Environment):

    def __init__(self):
        # self._observation_spec = specs.DiscreteArray(
        #     dtype=int, num_values=10, name="observation")
        # self._action_spec = specs.DiscreteArray(
        #     dtype=int, num_values=10, name="action")
        self._reset_next_step = True

    def step(self, action) -> TimeStep:
        if self._reset_next_step:
            return self.reset()
        horses = ["snickers", "mars", "malteaser"]
        winner = np.random.choice(horses, p=[0.1,0.3,0.6])
        pick = horses[action]

        print(f'Bet: {pick} Winner: {winner}')
        if pick == winner:
            if winner == "malteaser":
                # Expected value 1.66
                reward = 1
            elif winner =="mars":
                # Expected value 3.33
                reward = 4
            else:
                # Expected value 10
                reward = 10
        else:
            reward = -1
        self._reset_next_step = True
        return dm_env.termination(reward=reward, observation=self._observation())


    def observation_spec(self):
        return specs.Array((1,),
            dtype=int, name="observation")

    def action_spec(self):
        return specs.DiscreteArray(
            dtype=int, num_values=3, name="action")

    def reset(self) -> TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation = self._observation()
        return dm_env.restart(observation)

    def _observation(self):
        return 0