import numpy as np
from Envs import cliffwalking
from matplotlib import pyplot as plt

#env = cliffwalking.CliffWalking()

class TDESarsaAgent:

    class Policy:

        def __init__(self,num_actions,epsillon = 0.05):
            self._num_actions = num_actions
            self._pi = {}
            self._actions = [i for i in range(self._num_actions)]
            self._epsillon = epsillon

        def get_action(self,observation):
            if observation in self._pi.keys():
                return np.random.choice(self._actions,p=self._pi[observation])
            else:
                self._pi[observation] = (1/self._num_actions)*np.ones(self._num_actions,dtype=np.float32)
                return np.random.choice(self._actions, p=self._pi[observation])

        def update_policy(self,Q = {}):
            for state in Q.keys():
                best_action = np.argmax(Q[state])
                pi = (self._epsillon/self._num_actions)*np.ones(self._num_actions,dtype=np.float32)
                pi[best_action] += 1 - self._epsillon
                self._pi[state] = pi
            self._epsillon = 0.7*self._epsillon

        def __getitem__(self, item):
            if item[0] not in self._pi.keys():
                self._pi[item[0]] = (1 / self._num_actions) * np.ones(self._num_actions, dtype=np.float32)
            return self._pi[item[0]][item[1]]

    def __init__(self,environment, gamma = 0.85,alpha = 0.1):
        self._env = environment
        self._Q = {}
        self._num_actions = self._env.num_actions
        self._policy = self.Policy(self._num_actions)
        self._alpha = alpha
        self._gamma = gamma

    def update_q_Qlearning(self,last_state,last_action,next_state,reward):
        if last_state not in self._Q.keys():
            self._Q[last_state] = np.zeros(self._num_actions,dtype = np.float32)
        if next_state not in self._Q.keys():
            self._Q[next_state] = np.zeros(self._num_actions,dtype = np.float32)
        self._Q[last_state][last_action] += self._alpha * ((reward + self._gamma * np.max([self._Q[next_state][i] * self._policy[(next_state, i)] for i in range(self._num_actions)])) - self._Q[last_state][last_action])


    def update_q_expected_sarsa(self, last_state,last_action,next_state, reward):
        if last_state not in self._Q.keys():
            self._Q[last_state] = np.zeros(self._num_actions,dtype = np.float32)
        if next_state not in self._Q.keys():
            self._Q[next_state] = np.zeros(self._num_actions,dtype = np.float32)
        self._Q[last_state][last_action] += self._alpha*((reward + self._gamma*np.sum([self._Q[next_state][i]*self._policy[(next_state,i)] for i in range(self._num_actions)])) - self._Q[last_state][last_action])


    def _train_expected_sarsa(self,num_episodes = 100):
        avg_rewards = []
        for i in range(num_episodes):
            prev_state = self._env.start()
            action = self._policy.get_action(prev_state)
            is_done = False
            steps = 0
            r = 0
            while (not is_done) and (steps < 100) :
                if i == num_episodes-1:
                    print("Step " + str(steps))
                    self._env.render()
                reward,next_state,is_done,info = self._env.step(action)
                r += reward
                self.update_q_expected_sarsa(last_state=prev_state,last_action=action,next_state=next_state,reward=reward)
                steps += 1
                prev_state = next_state
                action = self._policy.get_action(prev_state)
                self._policy.update_policy(Q=self._Q)
            print("Steps to finish " + str(i) +" episode : " + str(steps))
            avg_rewards.append(np.log(((-9 - r)**2) + 0.00001))
        print(avg_rewards[-5:-1])
        plt.plot(avg_rewards)
        plt.show()

if __name__ == "__main__":
    size = (10, 10)
    cliffmask = np.zeros((10, 10))
    cliffmask[(0, 0, 0, 3, 3, 3, 3), (1, 2, 3, 4, 5, 6, 7)] = 1
    cfenv = cliffwalking.CliffWalking(cliff_mask=cliffmask, goal_state=(0, 9), initial_state=(0, 0), size=size)
    cfenv.render()
    agent = TDESarsaAgent(environment=cfenv)
    agent._train_expected_sarsa(num_episodes=500)


