import numpy as np
import data_utils as du
import math
import matplotlib.pyplot as plt

class Clock:

    def __init__(self):
        self._timestep = 0

    def tick(self):
        self._timestep += 1

    def get_time(self):
        return self._timestep

class Bandit:

    def __init__(self, policy = "epsilon-greedy",epsilon = 0.4):
        self._estimates = None
        self._num_plays = None
        self._policy = policy
        self._epsilon = epsilon
        self._last_action = None
        self._clock = None

    def init_estimates(self,estimates, clock):
        self._estimates = estimates
        self._num_plays = np.zeros_like(estimates,dtype=np.int64)
        self._clock = clock

    def generate_action(self):
        if self._policy == "epsilon-greedy":
            p = np.random.uniform(0,1)
            if p > self._epsilon:
                action = np.argmax(self._estimates)
                self._last_action = action
                self._epsilon -= 0.05*self._epsilon
                return action
            else:
                action = np.random.randint(0,len(self._estimates))
                self._last_action = action
                self._epsilon -= 0.05 * self._epsilon
                return action
        if self._policy == "softmax":
            softmax_extimates = np.zeros_like(self._estimates)
            exp_sum = np.sum(np.exp(self._estimates))
            for i in range(len(self._estimates)):
                softmax_extimates[i] = np.exp(self._estimates[i])/exp_sum
            action = np.argmax(softmax_extimates)
            self._last_action = action
            return action
        if self._policy == "UCB1":
            for i in range(len(self._num_plays)):
                if self._num_plays[i] == 0:
                    self._last_action = i
                    return i
            ucb_estimates = np.zeros_like(self._estimates)
            for i in range(len(self._estimates)):
                ucb_estimates[i] = self._estimates[i] + (2*math.sqrt((2*np.log(self._clock.get_time()))/self._num_plays[i]))
            action = np.argmax(ucb_estimates)
            self._last_action = action
            return action

    def update_estimates(self, reward):
        self._num_plays[self._last_action] += 1
        self._estimates[self._last_action] += (1/self._num_plays[self._last_action])*(reward - self._estimates[self._last_action])

class Casino:

    def __init__(self, bandit = Bandit(), type = "stationary", num_rounds = 10000):
        self._type = type
        self._clock = Clock()
        self._bandit = bandit
        self._num_rounds = num_rounds
        self._rewards = []
        self._regrets = []

    def start_game(self):
        q,q_true,best_ex_reward = du.generate_action_set(10,0,100)
        self._bandit.init_estimates(q,self._clock)
        for i in range(self._num_rounds):
            action = self._bandit.generate_action()
            print(action)
            rgen = q_true[action]
            print(rgen)
            print(best_ex_reward)
            reward = rgen.sample_reward()
            self._clock.tick()
            self._rewards.append(reward)
            regret = best_ex_reward*self._clock.get_time() - np.sum(self._rewards)
            self._regrets.append(regret)

            self._bandit.update_estimates(reward)

    def plot_regret(self):
        plt.plot(self._regrets)
        plt.show()

if __name__ == "__main__":

    b = Bandit(policy="UCB1")
    c = Casino(bandit=b)
    c.start_game()
    c.plot_regret()

