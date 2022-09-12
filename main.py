# reinforcement learning
import acme
from acme import types
from acme.wrappers import gym_wrapper
from acme.environment_loop import EnvironmentLoop
from acme.utils.loggers import TerminalLogger, InMemoryLogger

import test_env
from q_agent import QLearningAgent
# environments
import gym
import dm_env

# other
import numpy as np



def main():
    # env = acme.wrappers.GymWrapper(gym.make('Blackjack-v1'))
    env = test_env.TestEnv()
    # env = acme.wrappers.SinglePrecisionWrapper(env)

    # print env specs
    # env_specs = env.observation_space, env.action_space, env.reward_range  # env.observation_spec()
    # print('Observation Spec:', env.observation_space)
    # print('Action Spec:', env.action_space)
    # print('Reward Spec:', env.reward_range)
    # make first observation

    agent = QLearningAgent()


    for i in range(10):
        timestep = env.reset()
        agent.observe_first(timestep)
        # run an episode
        while not timestep.last():
            # generate an action from the agent's policy and step the environment
            action = agent.select_action(timestep.observation)
            timestep = env.step(action)
            agent.observe(action, next_timestep=timestep)
            if not timestep.last():
                # have the agent observe the timestep and let the agent update itself
                agent.update()

if __name__ == '__main__':
    main()