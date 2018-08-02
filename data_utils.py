import numpy as np

class RewardGenerator:

    def __init__(self,mu,sigma):
        self._mu = mu
        self._sigma = sigma

    def sample_reward(self):
        return np.random.normal(self._mu,self._sigma)

def generate_action_set(N, min_ex_reward = 0, max_ex_reward = 100):
    '''
    Generate a state set given the cardinality
    :param N: Number of possible states
    :return: A list Q --> Current Estimates of expected returns/payoff for that action
             A list Q_true --> List of tuples (mu,sig) where mu is the true mean of the return/payoff for that action
    '''

    Q = np.zeros(N)
    Q_true = []
    mus = np.random.uniform(low = min_ex_reward, high = max_ex_reward,size = N)
    best_ex_reward = np.max(mus)
    for i in range(N):
        r_g = RewardGenerator(mu=mus[i],sigma=1.0)
        Q_true.append(r_g)

    return Q, Q_true, best_ex_reward

if __name__ == "__main__":

    q,q_t,best_mu = generate_action_set(100,min_ex_reward=20,max_ex_reward=50)
    print(q)
    print(q_t)
    print(best_mu)
