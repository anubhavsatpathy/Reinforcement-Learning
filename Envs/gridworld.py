import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class GridWorld:

    def __init__(self):

        self._max_x = 4
        self._max_y = 4
        self._current_state = None
        self._num_states = self._max_x*self._max_y
        self._grid = np.zeros((self._max_x,self._max_y))
        self._terminal_states_x = (0,3)
        self._terminal_states_y = (0,3)
        self._grid[[self._terminal_states_x,self._terminal_states_y]] = 1
        self._state_space = [i for i in range(self._num_states)]
        self._action_space = [0,1,2,3]
        self._start_states = []
        for i in range(self._max_x):
            for j in range(self._max_y):
                append = True
                for k in range(len(self._terminal_states_x)):
                    if (i,j) == (self._terminal_states_x[k],self._terminal_states_y[k]):
                        append = False
                        break
                if append:
                    self._start_states.append((i,j))

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_actions(self):
        return len(self._action_space)

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space

    def start(self):
        idx = np.random.randint(0,len(self._start_states))
        start_state = self._start_states[idx]
        self._current_state = start_state
        return start_state

    def step(self, action):
        '''

        :param action:
        :return: (reward,next_state,is_done,info)
        '''
        state = self._current_state

        if action == UP:
            new_state = (state[0],state[1] + 1)
        if action == DOWN:
            new_state = (state[0],state[1] - 1)
        if action == RIGHT:
            new_state = (state[0] + 1,state[1])
        if action == LEFT:
            new_state = (state[0] - 1,state[1])

        if new_state[0] >= self._max_x or new_state[0] < 0 or new_state[1] >= self._max_y or new_state[1] < 0:
            return (-1,state,False,"Trying to go outside grid")

        for i in range(len(self._terminal_states_x)):
            if new_state[0] == self._terminal_states_x[i] and new_state[1] == self._terminal_states_y[i]:
                self._current_state = new_state
                return (1,new_state,True,"Congratulations - You have arrived")

        self._current_state = new_state
        return (-1,new_state,False,"You're on your way")

    def render(self):
        grid = np.array(self._grid)
        grid[self._current_state] = 1
        print(grid)

if __name__ == "__main__":

    env = GridWorld()
    actions = env.action_space
    state = env.start()
    env.render()
    is_done = False
    while not is_done:

        action = np.random.choice(actions)
        rew,ns,is_done,info = env.step(action)
        env.render()
        print(info)


