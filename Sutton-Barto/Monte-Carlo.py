import numpy as np
from Envs import gridworld

'''
Assumptions:
1. Actions encoded as integers from 0 to num_actions - 1
2. Discrete and finite action space
'''
grid = gridworld.GridWorld()

class MonteCarloAgent:

    class Policy:

        def __init__(self,num_actions):
            '''

            :param num_actions: The maximum number of actions possible for any state of the environment
            pi : The actual policy disc that contains state :[prob[action i]]
            '''
            self._num_actions = num_actions
            self._action_space = [i for i in range(num_actions)]
            self._type = "stochiastic"
            self._epsillon = 0.05
            self._pi = {}

        def get_action(self,observation):
            '''
            Returns the action sampled from pi
            :param observation: The state presented by the environment
            :return: action to be takes as sampled from the current policy
            '''
            if observation in self._pi.keys():
                return np.random.choice(self._action_space,p=self._pi[observation])
            else:
                self._pi[observation] = (1/self._num_actions)*np.ones(self._num_actions,dtype=np.float32)
                return np.random.choice(self._action_space,p=self._pi[observation])

        def update(self,Q = {}):
            '''
            Updates self._pi according to the latest action values estimated by the agent
            :param Q: A dictionary of the format {state : [action_value_for_action_i]}
            :return: None
            '''
            for state in Q.keys():
                best_action = np.argmax(Q[state])
                pi = (self._epsillon/self._num_actions)*np.ones(self._num_actions,dtype=np.float32)
                pi[best_action] += 1 - self._epsillon
                self._pi[state] = pi

    def __init__(self,env = grid):
        self._env = env
        self._action_space = env.action_space
        self._num_actions = len(self._action_space)
        self._policy = self.Policy(self._num_actions)
        self._Q = {}
        self._counts = {}
        self._sum_of_returns = {}
        self._num_steps = []

    def play(self):
        prev_state = self._env.start()
        is_done = False
        while not is_done:
            action = self._policy.get_action(prev_state)
            reward,next_state,is_done,info = self._env.step(action)
            self._env.render()
            prev_state = next_state

    def train(self,epochs = 10):
        for i in range(epochs):
            self._train_one_update(num_episodes=50)
            print("After" + str(i) + " updates, num_steps per episode : " + str(np.mean(self._num_steps)))

    def _train_one_update(self,num_episodes, max_steps_per_episode = 50,discount_factor = 0.85):
        '''
        Does GPI by estimating action values and updating policy
        :param num_episodes: The number of episoves generated per policy update
        :return: None
        '''
        for i in range(num_episodes):
            #print("Running Episode " + str(i))
            episode = []
            prev_state = self._env.start()
            #print(prev_state)
            is_done = False
            steps = 0
            G = {}
            while not is_done:   # Generate One episode
                action = self._policy.get_action(prev_state)
                reward,next_state,is_done,info = self._env.step(action)
                episode.append((prev_state,action,reward))
                steps += 1
                prev_state = next_state
                if steps > max_steps_per_episode:
                    is_done = True
            self._num_steps.append(steps)
            #print(episode)
            #print(steps)
            for i in range(len(episode)): # Calculate Discounted Returns for each state action pair encountered during episode

                state,action,r = episode[i]
                if (state,action) not in G.keys():
                    discounted_return = 0
                    for j in range(i,len(episode)):
                        reward = episode[j][2]
                        discounted_return += reward*(discount_factor**(j-i))
                    G[(state,action)] = discounted_return

            for sa_pair in G.keys(): # Update counts and sum of returns
                state = sa_pair[0]
                action = sa_pair[1]
                if state not in self._counts.keys():
                    self._counts[state] = np.ones(self._num_actions)
                    self._counts[state][action] += 1
                else:
                    self._counts[state][action] += 1
                if state not in self._sum_of_returns.keys():
                    self._sum_of_returns[state] = np.zeros(self._num_actions)
                    self._sum_of_returns[state][action] = G[(sa_pair)]
                else:
                    self._sum_of_returns[state][action] = G[(sa_pair)]

        for state in self._counts.keys():
            self._Q[state] = self._sum_of_returns[state]/self._counts[state]

        self._policy.update(self._Q)


if __name__ == "__main__":


    g_env = gridworld.GridWorld()
    agent = MonteCarloAgent(env=g_env)
    agent.train()
    agent.play()



