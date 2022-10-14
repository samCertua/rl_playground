# reinforcement learning
import acme
from acme import types
from acme.wrappers import gym_wrapper
from acme.environment_loop import EnvironmentLoop
from acme.utils.loggers import TerminalLogger, InMemoryLogger
import test_env
from q_agent import QLearningAgent
from deep_q_agent import DeepQLearningAgent
from monte_carlo_policy_gradient import MonteCarlo
import coin_flip
import test_env_addition
# environments
import catch
import catch_1d
import gym
import dm_env

# other
import numpy as np



def main():
    env = acme.wrappers.GymWrapper(gym.make('CartPole-v1'))
    # env = test_env.TestEnv()
    # env = test_env_addition.TestEnvAddition()
    # env = catch_1d.Catch(rows=4, columns=3)
    # env = coin_flip.CoinFlip()
    # env = acme.wrappers.SinglePrecisionWrapper(env)

    # print env specs
    # env_specs = env.observation_space, env.action_space, env.reward_range  # env.observation_spec()
    # print('Observation Spec:', env.observation_space)
    # print('Action Spec:', env.action_space)
    # print('Reward Spec:', env.reward_range)
    # make first observation

    # agent = QLearningAgent(env_specs=acme.specs.make_environment_spec(env))
    # agent = QLearningAgent(q=(30,5,5))
    agent = MonteCarlo(env)
    # agent = DeepQLearningAgent(q=(4,2))
    # agent = acme.agents.


    for i in range(50000):
        print("EPOCH "+str(i))
        timestep = env.reset()
        agent.observe_first(timestep)
        # run an episode
        while not timestep.last():
            # generate an action from the agent's policy and step the environment
            action = agent.select_action(timestep.observation)
            timestep = env.step(action)
            agent.observe(action, next_timestep=timestep)
            agent.update()
            # if timestep.last():
                # have the agent observe the timestep and let the agent update itself
                # agent.update()
        print(agent.total_reward)
    for i in range(1000):
        print("EPOCH "+str(i))
        timestep = env.reset()
        agent.observe_first(timestep)
        # run an episode
        while not timestep.last():
            # generate an action from the agent's policy and step the environment
            action = agent.select_action(timestep.observation, deployment=True)
            timestep = env.step(action)

            agent.observe(action, next_timestep=timestep)
            agent.update()
    return

if __name__ == '__main__':
    main()