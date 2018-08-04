import numpy as np
from Envs import cliffwalking,cartpole
from collections import deque
from matplotlib import pyplot as plt
import pprint

size = (10,10)
cliffmask = np.zeros((10,10))
cliffmask[(0,0,0,3,3,3,3),(1,2,3,4,5,6,7)] = 1
cfenv = cliffwalking.CliffWalking(cliff_mask=cliffmask,goal_state=(0,9),initial_state=(0,0),size=size)
cpenv = cartpole.CartPole()

class TD_n:

    class Policy:

        def __init__(self,num_actions,epsillon = 0.4):
            self._num_actions = num_actions
            self._pi = {}
            self._actions = [i for i in range(self._num_actions)]
            self._epsillon = epsillon
            self._decay = 0.75

        def decay_exploration(self):
            self._epsillon = self._decay*self._epsillon

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

        def __getitem__(self, item):
            if item[0] not in self._pi.keys():
                self._pi[item[0]] = (1 / self._num_actions) * np.ones(self._num_actions, dtype=np.float32)
            return self._pi[item[0]][item[1]]

    def __init__(self,env=cpenv,n = 5,alpha = 0.4,gamma = 1.0):
        self._env = env
        self._Q = {}
        self._n = n
        self._alpha = alpha
        self._min_alpha = 0.05
        self._decay = 0.5
        self._gamma = gamma
        self._num_actions = env.num_actions
        self._policy = self.Policy(self._num_actions)

    def update_q(self,episode,next_state,is_done):
        if next_state not in self._Q.keys():
            self._Q[next_state] = np.zeros(self._num_actions,dtype = np.float32)
        if is_done:
            while(len(episode) > 0):
                update = episode.popleft()
                state = update[0]
                if state not in self._Q.keys():
                    self._Q[state] = np.zeros(self._num_actions,dtype = np.float32)
                action = update[1]
                imm_reward = update[2]
                G = imm_reward + np.sum([(self._gamma**(i+1))*episode[i][2] for i in range(len(episode))]) + np.max([self._Q[next_state][j]*self._policy[(next_state,j)] for j in range(self._num_actions)])
                self._Q[state][action] += self._alpha*(G - self._Q[state][action])
                self._alpha = max(self._min_alpha,self._decay*self._alpha)
                return
        update = episode.popleft()
        state = update[0]
        if state not in self._Q.keys():
            self._Q[state] = np.zeros(self._num_actions, dtype=np.float32)
        action = update[1]
        imm_reward = update[2]
        G = imm_reward + np.sum([(self._gamma ** (i + 1)) * episode[i][2] for i in range(len(episode))]) + np.max(
            [self._Q[next_state][j] * self._policy[(next_state, j)] for j in range(self._num_actions)])
        self._Q[state][action] += self._alpha * (G - self._Q[state][action])
        self._alpha = max(self._min_alpha,self._decay*self._alpha)
        return

    def _collect_matrices(self):
        pass


    def _train(self,num_episodes):

        for i in range(num_episodes):
            episode = deque()
            steps = 0
            prev_state = self._env.start()
            is_done = False
            while (not is_done) and (steps < 1000):
                self._policy.decay_exploration()
                if i == num_episodes-1:
                    self._policy._epsillon = 0.0
                    #print("Step " + str(steps))
                    self._env.render()
                action = self._policy.get_action(prev_state)
                reward,next_state,is_done,info = self._env.step(action)
                steps += 1
                episode.append((prev_state,action,reward))
                if len(episode) == self._n or is_done:
                    self.update_q(episode,next_state,is_done)
                    self._policy.update_policy(self._Q)
                prev_state = next_state
            print("steps for episode {} : {}".format(i,steps))


if __name__ == "__main__":

    agent = TD_n(n=1)
    agent._train(1000)



