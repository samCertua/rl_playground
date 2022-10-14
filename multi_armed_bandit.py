import random

class MultiArmedBandit:

    def __init__(self, n: int):
        self._reset_next_step = True
        self.bandit_likelihoods = [random.random() for i in range(n)]
        self.bandit_pull_counts = [0 for i in range(n)]
        print("Bandit probs: "+str(self.bandit_likelihoods))

    def step(self, action):
        self.bandit_pull_counts[action]+=1
        print("Pulled bandit "+str(action) + f' ({self.bandit_likelihoods}) ({self.bandit_pull_counts}')
        if self._reset_next_step:
            return self.reset()
        self._reset_next_step = True
        if random.random() > self.bandit_likelihoods[action]:
            reward = 1
        else:
            reward = 0
        # print("Reward: "+str(reward))
        return self.observe(), reward, True

    def observe(self):
        return 0,0

    def reset(self):
        self._reset_next_step = False
        return self.observe(), None, False