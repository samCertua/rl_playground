import random

from journey_simulator import Journey
from user_simulator import *
from random_journey_agent_simulator import RandomJourneyAgent
from monte_carlo_journey_agent import MonteCarloJourneyAgent
from ppo_journey_agent import PPOJourneyAgent


def main():
    old_man = OldMan()
    young_woman = YoungWoman()
    inconsistent_man = InconsistentMan()
    users = [old_man, young_woman, inconsistent_man]
    env = Journey(random.choice(users))
    agent = RandomJourneyAgent(action_space=2, observation_space=None)
    agent = MonteCarloJourneyAgent(action_space=3, observation_space=None)
    agent = PPOJourneyAgent(action_space=3, observation_space=None)

    for i in range(10000):
        print("EPOCH "+str(i) + " User: "+str(env.user.__class__))
        observation, reward, last = env.reset()
        agent.observe_first(observation)
        # run an episode
        while not last:
            # generate an action from the agent's policy and step the environment
            action = agent.select_action(observation)
            observation, reward, last = env.step(action)
            agent.observe(action, reward, observation, last)
            agent.update()
            # if timestep.last():
                # have the agent observe the timestep and let the agent update itself
                # agent.update()
        env.user = random.choice(users)
    pass

if __name__ == '__main__':
    main()