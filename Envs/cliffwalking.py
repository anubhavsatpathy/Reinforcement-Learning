import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class CliffWalking():

    def __init__(self,cliff_mask,goal_state,initial_state,size=(10,10)):
        self._max_x = size[0]
        self._max_y = size[1]
        self._grid = [self._max_x*['_'] for i in range(self._max_y)]
        self._cliff_mask = cliff_mask
        it = np.nditer(self._cliff_mask,flags = ['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if self._cliff_mask[idx[0],idx[1]] == 1:
                self._grid[idx[0]][idx[1]] = 'C'
            it.iternext()
        self._goal_state = goal_state
        self._start_state = initial_state
        self._grid[self._goal_state[0]][self._goal_state[1]] = 'G'
        self._num_actions = 4
        self._actions = [UP,DOWN,LEFT,RIGHT]
        self._current_state = None

    @property
    def num_actions(self):
        return self._num_actions

    def start(self):
        self._current_state = self._start_state
        self._update_current_state(self._start_state)
        return self._current_state

    def step(self,action):

        state = self._current_state

        if action == UP:
            new_state = (state[0], state[1] + 1)
        if action == DOWN:
            new_state = (state[0], state[1] - 1)
        if action == RIGHT:
            new_state = (state[0] + 1, state[1])
        if action == LEFT:
            new_state = (state[0] - 1, state[1])

        if new_state[0] >= self._max_x or new_state[0] < 0 or new_state[1] >= self._max_y or new_state[1] < 0:
            return (-2,state,False,"Trying to go outside grid")

        if self._cliff_mask[new_state[0],new_state[1]] == 1:
            self._update_current_state(self._start_state)
            return (-100,self._current_state,False,"You jumped off a cliff!!")

        if new_state == self._goal_state:
            self._update_current_state(self._goal_state)
            return (10,self._current_state,True,"You've Reached")

        self._update_current_state(new_state)
        return (-1,self._current_state,False,"Youre on your way")

    def _update_current_state(self,state):
        self._grid[self._current_state[0]][self._current_state[1]] = '_'
        self._grid[state[0]][state[1]] = 'I'
        self._current_state = state

    def render(self):
        for rown in self._grid:
            print(rown)

if __name__ == "__main__":

    cliff_mask = np.zeros((5,5))
    cliff_mask[(2,2),(2,3)] = 1
    env = CliffWalking(cliff_mask,(4,4),(0,0),size=(5,5))
    env.start()
    for i in range(5):
        print("After step " + str(i))
        env.render()
        action = np.random.choice([UP,DOWN,RIGHT,LEFT])
        env.step(action)
