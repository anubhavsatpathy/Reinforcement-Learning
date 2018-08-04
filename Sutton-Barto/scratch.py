import gym

e = gym.make('CartPole-v0')
e.reset()
while True:
    action = e.action_space.sample()
    ns,r,d,i = e.step(action)
    print(int(not d))
    if d:
        break
