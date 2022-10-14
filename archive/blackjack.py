
import acme
import dm_env
from dm_env._environment import TimeStep


class Blackjack(dm_env.Environment):

    def step(self, action) -> TimeStep:
        pass

    def observation_spec(self):
        pass

    def action_spec(self):
        pass

    def reset(self) -> TimeStep:
        pass