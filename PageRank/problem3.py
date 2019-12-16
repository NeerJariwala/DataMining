import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 3: PageRank algorithm (version 1) 
    In this problem, we implement a simplified version of the pagerank algorithm, which doesn't consider sink node problem or sink region problem.
'''

#--------------------------
def compute_P(A):
    '''
        compute the transition matrix P from adjacency matrix A. P[j][i] represents the probability of moving from node i to node j.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                   The values in each column of matrix P should sum to 1.
    '''
    P = A / A.sum(axis=0)
    return P



#--------------------------
def random_walk_one_step(P, x_i):
    '''
        compute the result of one step random walk.
        Input: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                x_i: pagerank scores before the i-th step of random walk. a numpy vector of length n.
        Output: 
                x_i_plus_1: pagerank scores after the i-th step of random walk. a numpy vector of length n.
    '''
    x_i_plus_1 = P.dot(x_i)
    return x_i_plus_1


#--------------------------
def random_walk(P, x_0, max_steps=10000):
    '''
        Compute the result of multiple-step random walk. 
        The random walk should stop if the score vector x no longer change (converge) after one step of random walk, or the number of iteration reached max_steps.
        Input: 
                P: transition matrix, a (n by n) numpy matrix of float values.  P[j][i] represents the probability of moving from node i to node j.
                x_0: the initial pagerank scores. a numpy vector of length n.
                max_steps: the maximum number of random walk steps. an integer value.  
        Output: 
                x: the final pagerank scores after multiple steps of random walk. a numpy vector of length n.
                n_steps: the number of steps of random walk actually used (for example, if the vector x no longer changes after 3 steps of random walk, return the value 3. 
     '''
    old_x = x_0
    x = random_walk_one_step(P, x_0)
    n_steps = 1
    for i in range(1, max_steps):
        if np.allclose(old_x, x):
            print(x)
            break
        else:
            old_x = x
            x = random_walk_one_step(P, x)
            n_steps = n_steps + 1
    return x, n_steps


#--------------------------
def pagerank_v1(A):
    ''' 
        A simplified version of PageRank algorithm.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Here we ignore the issues of sink nodes and sink regions in the network.
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the pagerank scores, a numpy vector of float values, such as np.array([.3, .2, .5])
    '''
    # initialize the pagerank score vector
    num_nodes = A.shape[0] # get the number of nodes (n)
    x_0 =  np.ones(num_nodes)/num_nodes
    P = compute_P(A)
    x, n_steps = random_walk(P, x_0)

    return x
