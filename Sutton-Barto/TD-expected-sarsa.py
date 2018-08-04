import numpy as np
from Envs import cliffwalking
from matplotlib import pyplot as plt
from Envs import cartpole

#env = cliffwalking.CliffWalking()

class TDESarsaAgent:

    class Policy:

        def __init__(self,num_actions,epsillon = 0.3):
            self._num_actions = num_actions
            self._pi = {}
            self._actions = [i for i in range(self._num_actions)]
            self._epsillon = epsillon
            self._decay = 0.9
        def decay_exploration(self):
            self._epsillon = self._epsillon*self._decay

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
            #self._epsillon = 0.7*self._epsillon

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


    def _train_expected_sarsa(self,num_episodes = 500):
        for i in range(num_episodes):
            self._policy.decay_exploration()
            prev_state = self._env.start()
            action = self._policy.get_action(prev_state)
            is_done = False
            steps = 0
            while (not is_done) and (steps < 1000):
                if i == num_episodes-1:
                    print("Step " + str(steps))
                    self._env.render()
                reward,next_state,is_done,info = self._env.step(action)
                self.update_q_Qlearning(last_state=prev_state,last_action=action,next_state=next_state,reward=reward)
                steps += 1
                prev_state = next_state
                action = self._policy.get_action(prev_state)
                self._policy.update_policy(Q=self._Q)
            print("Steps to finish " + str(i) +" episode : " + str(steps))


if __name__ == "__main__":
    size = (10, 10)
    cliffmask = np.zeros((10, 10))
    cliffmask[(0, 0, 0, 3, 3, 3, 3), (1, 2, 3, 4, 5, 6, 7)] = 1
    cfenv = cliffwalking.CliffWalking(cliff_mask=cliffmask, goal_state=(0, 9), initial_state=(0, 0), size=size)
    cfenv.render()
    cpenv = cartpole.CartPole()
    agent = TDESarsaAgent(environment=cpenv)
    agent._train_expected_sarsa(num_episodes=500)


