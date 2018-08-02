import numpy as np

def policy_evaluate(transition_matrix, reward_matrix, policy_matrix,gamma,epsilon = 0.5):
    '''
    Returns v_pi[s] for s in range N <-- number of states in the MDP
    :param transition_matrix: An N*A*N matrix that denotes the trasition propabilities
    :param reward_matrix: A N*A matrix denoting the rewards for each state-action pair
    :param policy_matrix: A N*A stochiastic matrix denoting the policy
    :return: v_pi
    '''
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]

    v_pi = np.zeros(num_states)
    while(True):
        v_pi_old = v_pi
        for i in range(num_states):
            v = 0
            for j in range(num_actions):
                for k in range(num_states):
                    v += policy_matrix[i,j]*(reward_matrix[i,j] + gamma*transition_matrix[i][j][k]*v_pi_old[k])
            v_pi[i] = v

        if np.sum(np.abs(v_pi_old - v_pi)) < epsilon:
            return v_pi

def policy_improve(transition_matrix,reward_matrix,value_array,gamma):
    '''
    Creates and returns a better policy acting greedily w.r.t q[s][a]
    :param transition_matrix: An N*A*N matrix that denotes the trasition propabilities
    :param reward_matrix: A N*A matrix denoting the rewards for each state-action pair
    :param value_array: A N size array defining the value function
    :return: policy_matrix : A stochiastic N*A matrix defining a greedy policy
    '''
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]
    new_policy = np.zeros(num_states,num_actions)
    for i in range(num_states):
        q = np.zeros(num_actions)
        for j in range(num_actions):
            q[j] = reward_matrix[i][j]
            for k in range(num_states):
                q[j] += gamma*transition_matrix[i][j][k]*value_array[k]
        best_action = np.argmax(q)
        new_policy[i][best_action] = 1

    return new_policy
