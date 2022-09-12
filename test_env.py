
import acme
import dm_env
from dm_env._environment import TimeStep
from dm_env import specs


class TestEnv(dm_env.Environment):

    def __init__(self):
        self.time = 0
        self.magic_number = 1
        self._observation_spec = specs.DiscreteArray(
            dtype=int, num_values=10, name="observation")
        self._action_spec = specs.DiscreteArray(
            dtype=int, num_values=10, name="action")

    def step(self, action) -> TimeStep:
        self.time+=1

        if action==self.magic_number:
            reward = 1
        else:
            reward = -1
        if self.time==50:
            return dm_env.termination(reward=reward, observation=self._observation())
        else:
            return dm_env.transition(reward=reward, observation=self._observation())
        pass

    def observation_spec(self):
        return self.observation_spec()

    def action_spec(self):
        return self.action_spec()

    def reset(self) -> TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        observation = self.magic_number
        # Reset the diagnostic information.
        self._last_info = None
        return dm_env.restart(observation)

    def _observation(self):
        return self.magic_number