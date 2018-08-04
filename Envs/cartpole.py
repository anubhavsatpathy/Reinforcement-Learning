import gym
import numpy as np
import math

class CartPole:

    def __init__(self):
        self._env = gym.make('CartPole-v0')
        self._upper_bounds = [self._env.observation_space.high[0],0.5,self._env.observation_space.high[2],math.radians(50)/1]
        self._lower_bounds = [self._env.observation_space.low[0],-0.5,self._env.observation_space.low[2],-math.radians(50)/1]
        self._bin_x = np.linspace(self._lower_bounds[0],self._upper_bounds[0],2)
        self._bin_x_dot = np.linspace(self._lower_bounds[1],self._upper_bounds[1],2)
        self._bin_theta = np.linspace(self._lower_bounds[2],self._upper_bounds[2],6)
        self._bin_theta_dot = np.linspace(self._lower_bounds[3],self._upper_bounds[3],3)
        self._buckets = (1,1,6,12)

    def discretize(self,state):

        discrete_x = int(np.digitize(state[0],self._bin_x))
        discrete_x_dot = int(np.digitize(state[1],self._bin_x_dot))
        discrete_theta = int(np.digitize(state[2],self._bin_theta))
        discrete_theta_dot = int(np.digitize(state[3],self._bin_theta_dot))
        return (discrete_x,discrete_theta,discrete_theta_dot)
        #return (discrete_theta,discrete_theta_dot)


    @property
    def num_actions(self):
        return self._env.action_space.n

    def start(self):
        state = self.discretize(self._env.reset())
        return state

    def step(self,action):
        next_state,reward,done,info = self._env.step(action)
        discrete_state = self.discretize(next_state)
        return (int(not done),discrete_state,done,info)

    def render(self):
        self._env.render()
