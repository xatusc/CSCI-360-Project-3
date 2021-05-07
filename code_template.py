# code_template.py
#=============================================================
# Code template for project 3 - MDP and Reinforcement Learning

# Created by Shihan Lu, Feb 21, 2021
# If you find any mistakes about this template, please contact
# Shihan Lu (shihanlu@usc.edu) or post on Piazza.
#=============================================================


import argparse
import os
import gym
import numpy as np
from gym.envs.toy_text.frozen_lake import generate_random_map


def value_iteration(env, gamma=0.95, theta=0.0001): # Do not change arguments
    ''' Performs value iteration for the given environment

    :param env: Unwrapped Open AI environment
    :param gamma: Decay rate
    :param theta: Acceptable convergence rate

    :return: policy, a |State|x|Action| matrix of the probability
            of taking an action in a given state. Store it in the
            form of a 2d list.

            For the given parameters: gamma=0.95, theta=0.0001, you should get the following outputs:
            [[1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.5  0.   0.5  0.  ]
             [0.25 0.25 0.25 0.25]
             [0.   0.   0.   1.  ]
             [0.   1.   0.   0.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.25 0.25 0.25 0.25]
             [0.   0.   1.   0.  ]
             [0.   1.   0.   0.  ]
             [0.25 0.25 0.25 0.25]]

            The first element in the list [1. 0. 0. 0.] means that at
            the spot 0, the probability of moving left is 100%, the probabilities
            of moving right, up and down are all 0%.
    '''

    nS = env.nS # number of states
    nA = env.nA # number of actions
    envP = env.P # environment dynamics model

    V = np.zeros(nS) # initialize array of state values with all zeros. Do not change
    policy = np.ones([nS, nA]) / nA  # dummy policy which has 0.25 probability for each action at any state.
    delta = 0
    # Replace this policy with your implementation

    #==========================================
    "*** YOUR CODE HERE FOR VALUE ITERATION***"
    empty_policy = np.zeros([nS, nA]) 

    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            max_val = -9999999
            val_array = [0,0,0,0]
            for a in range(nA):
                x = envP[s][a]
                sum = 0
                for prob, next_state, reward, is_terminal in x:
                    val = prob * (reward + gamma * V[next_state])
                    sum += val
                if sum > max_val: 
                    max_val = sum
                val_array[a] = sum
            V[s] = max_val
            delta = max(delta, abs(v - V[s]))
            empty_policy[s] = val_array
        if delta < theta: 
            break

    for s in range(nS):
        best = np.amax(empty_policy[s])
        #Not hole or end#
        if best > 0:
            #Get all best policy#
            best_index = np.argwhere(empty_policy[s] >= best)
            for a in range(nA):
                #Max policy set to 1#
                if a in best_index:
                    policy[s][a] = 1/len(best_index)
                #Other policy set to 0#
                if a not in best_index:
                    policy[s][a] = 0


    #==========================================
    return policy  # Do not change


def policy_evaluation(policy, env, gamma=0.95, theta=0.0001): # Do not change arguments
    ''' Evaluate the state values following the given policy

    :param policy: 2D matrix |State|x|Action|, where each entry is
        the probability of which action to take in each state
    :param env: Unwrapped OpenAI gym environment
    :param gamma: Discount factor
    :param theta: Stopping condition
    :return List of state value. The size of list equals the number of states.

            For the given parameters: gamma=0.95, theta=0.0001, you should get the following outputs:
            [0.00752935 0.00673683 0.01418894 0.00639463 0.01019074 0.
             0.03249842 0.         0.02525507 0.07092393 0.12265383 0.
             0.         0.15073275 0.41302033 0.        ]
    '''

    nS = env.nS # number of states
    nA = env.nA # number of actions
    envP = env.P # environment dynamics model
    V = np.zeros(nS) # initialize array V[s]=0 for all s, do not change

    delta = 0
    #==========================================
    "*** YOUR CODE HERE FOR POLICY EVALUATION***"
    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            sum = 0
            for a in range(nA):
                x = envP[s][a]
                for prob, next_state, reward, is_terminal in x:
                    sum += policy[s][a] * prob * (reward + gamma * V[next_state])
            V[s] = sum
            delta = max(delta, abs(v - V[s]))
        if delta < theta: 
            break
 
    #==========================================
    return V # Do not change


def policy_iteration(env, gamma=0.95, theta=0.0001):  # do not change arguments
    ''' Perform policy iteration with initially equally likely actions

    Initialize each action to be equally likely for all states
    Policy improvement is run until a stable policy is found

    :param env: Unwrapped OpenAI gym environment
    :param gamma: discount factor for both policy evaluation and policy improvement
    :para theta: Stopping condition for both policy evaluation and policy improvement
    :return policy, a |State|x|Action| matrix of the probability
            of taking an action per state.

            For the given parameters: gamma=0.95, theta=0.0001, you should get the following outputs:
            [[1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.   0.   0.   1.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.5  0.   0.5  0.  ]
             [0.25 0.25 0.25 0.25]
             [0.   0.   0.   1.  ]
             [0.   1.   0.   0.  ]
             [1.   0.   0.   0.  ]
             [0.25 0.25 0.25 0.25]
             [0.25 0.25 0.25 0.25]
             [0.   0.   1.   0.  ]
             [0.   1.   0.   0.  ]
             [0.25 0.25 0.25 0.25]]
    '''

    nS = env.nS # number of states
    nA = env.nA # number of actions
    envP = env.P # environment dynamics model
    V = np.zeros(nS) # initialize array V[s]=0 for all s, do not change
    policy = np.ones([nS, nA]) / nA # initialize policy that each action to be equally likely for all states
    
    #==========================================
    "*** YOUR CODE HERE FOR POLICY ITERATION***"
    while True:
        V = policy_evaluation(policy, env)
        policy_stable = True
        for s in range(nS):
            old_action = np.argwhere(policy[s] >= np.round(np.amax(policy[s]), 10))
            max_val = -9999999
            state_val_array = [0,0,0,0]
            for a in range(nA):
                x = envP[s][a]
                sum = 0
                for prob, next_state, reward, is_terminal in x:
                    val = prob * (reward + gamma * V[next_state])
                    sum += val
                state_val_array[a] = sum
                if sum >= max_val: 
                    max_val = sum
            new_action = np.argwhere(np.round(state_val_array, 10) >= np.round(np.amax(state_val_array), 10))
            for a in range(nA):
                if a in new_action:
                    policy[s][a] = 1/len(new_action)
                #Other policy set to 0
                if a not in new_action:
                    policy[s][a] = 0    
            if not np.array_equal(old_action, new_action):
                policy_stable = False
        if policy_stable:
            break

    #==========================================
    return policy  # Do not change


def q_learning_1(env, alpha=0.5, gamma=0.95, epsilon=0.5, num_episodes=500):
    ''' Performs Q-learning version 1 for the given environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize a vanilla epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Epsilon in epsilon-greedy method
    :param num_episodes: Number of episodes to use for learning

    :return: Q table and policy

    For the given parameters: alpha=0.5, gamma=0.95, epsilon=0.1, num_episodes=500, you should get the following outputs
    Q table:
         [[0.        0.        0.        0.      ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.11875   0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.        0.       ]
         [0.        0.        0.2375    0.       ]
         [0.        0.        0.5       0.1128125]
         [0.        0.        0.        0.       ]]

    policy
         [[0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.   0.   1.   0.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]]
    '''        

    np.random.seed(42) # do not change
    nS = env.nS # number of states
    nA = env.nA # number of actions
    Q = np.zeros([nS, nA]) # initialize Q-table with zeros. Do not change
    policy = np.ones([nS, nA]) / nA # dummy policy that each action is equally likely, do not change

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 1***"
    #### ======= Vanilla epsilon-greedy
    for e in range(num_episodes):
        s = env.reset()
        while True:
            p = np.random.rand()
            action = 0
            if p < epsilon:
                #Take random action
                action = np.random.choice(4)
            else:
                #Take current best action
                action = np.argmax(Q[s])
            new_state, reward, done, info = env.step(action)
            Q[s][action] = Q[s][action] + alpha * (reward + gamma * max(Q[new_state]) - Q[s][action] )
            s = new_state
            if done:
                break
    
    #Policy extraction
    for s in range(nS):
        best = np.amax(Q[s])
        #Not hole or end#
        if best > 0:
            #Get all best policy#
            best_index = np.argwhere(Q[s] >= best)
            for a in range(nA):
                #Max policy set to 1#
                if a in best_index:
                    policy[s][a] = 1/len(best_index)
                #Other policy set to 0#
                if a not in best_index:
                    policy[s][a] = 0

    #==========================================
    return Q, policy  # do not change



def q_learning_2(env, alpha=0.5, gamma=0.95, epsilon_0=1, epsilon_min=0.01, Lambda=0.001, num_episodes=500):
    ''' Performs Q-learning version 2 for the given environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize a decay epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon_0: Initial epsilon
    :param epsilon_min: minimal epsilon
    :param Lambda: a constant in exponential calculation
    :param num_episodes: Number of episodes to use for learning

    :return: Q table and policy

    For the given parameters: alpha=0.5, gamma=0.95, epsilon_0=1.0, epsilon_min = 0.01, Lambda = 0.001, num_episodes=500,
    you should get the following outputs.
    Q-table:
        [[0.36713299 0.37985477 0.35843552 0.35581387]
         [0.17517206 0.15872685 0.19903197 0.36888308]
         [0.40467574 0.40275562 0.49238606 0.40385159]
         [0.22828427 0.07587396 0.2194184  0.42327732]
         [0.45935126 0.2309994  0.27355389 0.21700694]
         [0.         0.         0.         0.        ]
         [0.26192058 0.28668218 0.19145272 0.14202863]
         [0.         0.         0.         0.        ]
         [0.25148164 0.33003944 0.31387329 0.55814431]
         [0.23155422 0.65397369 0.40905852 0.30759947]
         [0.60206605 0.21343306 0.50034703 0.38379698]
         [0.         0.         0.         0.        ]
         [0.         0.         0.         0.        ]
         [0.16369181 0.71453194 0.69310478 0.37951278]
         [0.68626787 0.74187664 0.95385906 0.68487487]
         [0.         0.         0.         0.        ]]

    Policy:
        [[0.   1.   0.   0.  ]
         [0.   0.   0.   1.  ]
         [0.   0.   1.   0.  ]
         [0.   0.   0.   1.  ]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.   1.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.   0.   0.   1.  ]
         [0.   1.   0.   0.  ]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.   1.   0.   0.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]]
    '''

    np.random.seed(42) # do not change
    nS = env.nS # number of states
    nA = env.nA # number of actions
    Q = np.zeros([nS, nA]) # initialize Q-table with zeros, do not change
    policy = np.ones([nS, nA]) / nA # dummy policy that each action is equally likely, do not change

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 2***"
    #### ======= Decay threshold version epsilon-greedy
    for e in range(num_episodes):
        s = env.reset()
        epsilon = epsilon_0 * np.exp(-Lambda * e)
        if epsilon < epsilon_min:
            epsilon = epsilon_min
        while True:
            p = np.random.rand()
            action = 0
            if p < epsilon:
                #Take random action
                action = np.random.choice(4)
            else:
                #Take current best action
                action = np.argmax(Q[s])
            new_state, reward, done, info = env.step(action)
            Q[s][action] = Q[s][action] + alpha * (reward + gamma * max(Q[new_state]) - Q[s][action] )
            s = new_state
            if done:
                break
    
    #Policy extraction
    for s in range(nS):
        best = np.amax(Q[s])
        #Not hole or end#
        if best > 0:
            #Get all best policy#
            best_index = np.argwhere(Q[s] >= best)
            for a in range(nA):
                #Max policy set to 1#
                if a in best_index:
                    policy[s][a] = 1/len(best_index)
                #Other policy set to 0#
                if a not in best_index:
                    policy[s][a] = 0

    ## ========================================
    return Q, policy  # do not change



def q_learning_3(env, alpha=0.5, gamma=0.95, epsilon=0.5, num_episodes=500):
    ''' Performs Q-learning version 3 for the given environment

    Note:
    - Initialize Q-table to all zeros
    - Utilize an effective exploiting epsilon-greedy method for action selection

    :param env: Unwrapped Open AI environment
    :param alpha: Learning rate
    :param gamma: Discount factor
    :param epsilon: Epsilon in epsilon-greedy method
    :param num_episodes: Number of episodes to use for learning

    :return: Q table and policy,

    For the given parameters: alpha=0.5, gamma=0.95, epsilon=0.5, num_episodes=500, you should get the following outputs.
    Q table:
         [[0.21151246 0.21022286 0.25999264 0.20899564]
         [0.09993746 0.14133315 0.0705775  0.18931574]
         [0.14854259 0.1505181  0.14683159 0.14670341]
         [0.13463091 0.13786784 0.05830575 0.14820557]
         [0.31862173 0.27826314 0.23927896 0.09306707]
         [0.         0.         0.         0.        ]
         [0.09036269 0.0509784  0.01831764 0.02214529]
         [0.         0.         0.         0.        ]
         [0.10107658 0.09926206 0.31749132 0.39445481]
         [0.35796593 0.44074778 0.11963118 0.296104  ]
         [0.28219434 0.06914525 0.30646221 0.19680597]
         [0.         0.         0.         0.        ]
         [0.         0.         0.         0.        ]
         [0.09177565 0.2651842  0.51977341 0.62812079]
         [0.62003614 0.70870562 0.858487   0.49727083]
         [0.         0.         0.         0.        ]]

    policy:
         [[0.   0.   1.   0. ]
         [0.   0.   0.   1.  ]
         [0.   1.   0.   0.  ]
         [0.   0.   0.   1.  ]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [1.   0.   0.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.   0.   0.   1.  ]
         [0.   1.   0.   0.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]
         [0.25 0.25 0.25 0.25]
         [0.   0.   0.   1.  ]
         [0.   0.   1.   0.  ]
         [0.25 0.25 0.25 0.25]]
    '''

    np.random.seed(42) # do not change
    nS = env.nS # number of states
    nA = env.nA # number of actions
    Q = np.zeros([nS, nA]) # initialize Q-table with zeros, do not change
    policy = np.ones([nS, nA]) / nA # dummy policy that each action is equally likely, do not change

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING VERSION 3***"
    ##### ===== Effective exploiting epsilon-greedy q-learning
    for e in range(num_episodes):
        s = env.reset()
        while True:
            p = np.random.rand()
            action = 0
            if p > epsilon and np.sum(Q[s]) != 0:
                #Take current best action
                action = np.argmax(Q[s]) 
            else:
                #Take random action
                action = np.random.choice(4)
            new_state, reward, done, info = env.step(action)
            Q[s][action] = Q[s][action] + alpha * (reward + gamma * max(Q[new_state]) - Q[s][action] )
            s = new_state
            if done:
                break
    
    #Policy extraction
    for s in range(nS):
        best = np.amax(Q[s])
        #Not hole or end#
        if best > 0:
            #Get all best policy#
            best_index = np.argwhere(Q[s] >= best)
            for a in range(nA):
                #Max policy set to 1#
                if a in best_index:
                    policy[s][a] = 1/len(best_index)
                #Other policy set to 0#
                if a not in best_index:
                    policy[s][a] = 0

    #==========================================
    return Q, policy  # do not change


def get_discrete_state(state):  # Do not change this function
    '''
    Convert the continuous state to discrete state in CartPole
    :param state: state
    :return: discrete state
    '''
    np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])  # discretization step
    discrete_state = state / np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(np.int))


def q_learning_cart(env, alpha=0.2, gamma=1, epsilon_0=0.9, epsilon_min=0.001, Lambda=0.001, num_episodes=10000):  # add your own arguments based on your implementation
    ''' Performs Q-learning for the CartPole environment

    :param env: Unwrapped Open AI environment
    :return: Q table
    '''

    # only use np.random.rand() for random sampling, and seed the random sampling as the following line
    np.random.seed(42)  # do not change
    nA = env.action_space.n  # number of actions
    state_space = [30, 30, 50, 50]  # create the discrete state space. Do not change
    Q = np.random.uniform(low=0, high=1, size=(state_space + [nA]))  # initialize Q-table with random numbers
    # Q = np.zeros(state_space+[nA])  # you can also initialize Q with zeros

    #==========================================
    "*** YOUR CODE HERE FOR Q-LEARNING FOR CARTPOLE***"
    "Q-Learning 3"
    for e in range(num_episodes):
        env.reset()
        current_state = get_discrete_state(env.reset())
        epsilon = epsilon_0 * np.exp(-Lambda * e)
        if epsilon < epsilon_min:
            epsilon = epsilon_min
        while True:
            p = np.random.rand()
            action = 0    
            if p > epsilon and np.sum(Q[current_state]) != 0:
                #Take current best action
                action = np.argmax(Q[current_state])            
            else:
                #Take random action
                action = np.random.choice(nA)
            next_state, reward, done, _ = env.step(action)
            Q[current_state + (action,)] = Q[current_state + (action,)] + alpha * (reward + gamma * max(Q[get_discrete_state(next_state)]) - Q[current_state + (action,)] )
            current_state = get_discrete_state(next_state)
            if done:
                break
    ## ========================================
    return Q  # do not change


def compare_policies(policy1, policy2): # Do not change this function
    '''
    :param policy1
    :param policy2
    :return: the number of states which have different policies
    '''
    policy1, policy2 = np.array(policy1), np.array(policy2)
    if np.shape(policy1) != np.shape(policy2):
        print("Two policies are not in the same size. Cannot compare")
        return
    l = len(policy1)
    return np.sum([1 if not np.array_equal(policy1[i], policy2[i]) else 0 for i in range(l)])


if __name__ == "__main__":
    # Set parser
    parser = argparse.ArgumentParser(description='Select a particular question')
    parser.add_argument("-q", "--question",
                        type=str, choices=['1', '2', '3', '4', '4-1', '4-2', '4-3'],
                        help="selection a question to test")
    # add a visualization argument
    parser.add_argument("-vis", "--visualization",
                        help="visualize the lake and execution",
                        action="store_true")
    args = parser.parse_args()

    os.system('') # trick way to colorize the console for environment rendering
    np.set_printoptions(suppress=True)
    np.random.seed(42) # seed numpy - do not change this line
    # random_map = generate_random_map(size=10, p=0.6) # generate a random map
    custom_map = [ # customize a map
        'SFFF',
        'FHFH',
        'FFFH',
        'HFFG'
    ]
    env = gym.make('FrozenLake-v0', desc=custom_map)
    env.seed(42)  # seed env - do not change this line
    print('Environment was loaded!\n')

    #====== Useful functions to help your coding and visualize the environment
    if args.visualization:
        print('Environment and execution visualization:')
        env.reset() # reset the environment
        env.render() # render the environment

        random_action = np.random.choice(env.nA) # take a random action
        new_state, reward, done, info = env.step(random_action) # execute the action
        # print(new_state, reward, done, info)
        env.render() # render the environment again after taking the action

        print('\nReset state:')
        env.reset()
        env.render()
        print("\n")
    # ========================

    # Driver codes to test the functions
    "*** Task 1: use value iteration to find the policy ***"
    if args.question == '1':
        print("Test task 1 - value iteration\n")
        policy1 = value_iteration(env)
        print(policy1)

    "*** Task 2: use policy evaluation to find the state values ***"
    if args.question == '2':
        print("Test task 2 - policy evaluation\n")
        policy2 = np.ones([env.nS, env.nA]) / env.nA  # policy that each action is equally likely
        value2 = policy_evaluation(policy2, env)
        print(value2)

    "*** Task 3: use policy iteration to find the policy ***"
    if args.question == '3':
        print("Test task 3 - policy iteration\n")
        policy3 = policy_iteration(env)
        print(policy3)

    "*** Task 4: use Q-Learning to learn the Q table and policy***"
    if args.question == '4-1':
        print("Test task 4 - q-learning version 1")
        q_table, policy = q_learning_1(env)
        print('\n q_table from q-learning version 1:\n', q_table)
        print('\n policy from q-learning version 1:\n', policy)

    if args.question == '4-2':
        print("Test task 4 - q-learning version 2")
        q_table, policy = q_learning_2(env)
        print('\n q_table from q-learning version 2:\n', q_table)
        print('\n policy from q-learning version 2:\n', policy)

    if args.question == '4-3':
        print("Test task 4 - q-learning version 3")
        q_table, policy = q_learning_3(env)
        print('\n q_table from q-learning version 3:\n', q_table)
        print('\n policy from q-learning version 3:\n', policy)