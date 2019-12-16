import numpy as np
from problem3 import compute_P,random_walk

#-------------------------------------------------------------------------
'''
    Problem 4: Solving sink-node problem in PageRank
    In this problem, we implement the pagerank algorithm which can solve the sink node problem.
'''

#--------------------------
def compute_S(A):
    '''
        compute the transition matrix S from addjacency matrix A, which solves sink node problem by filling the all-zero columns in A.
        S[j][i] represents the probability of moving from node i to node j.
        If node i is a sink node, S[j][i] = 1/n.
        Input: 
                A: adjacency matrix, a (n by n) numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                S: transition matrix, a (n by n) numpy matrix of float values.  S[j][i] represents the probability of moving from node i to node j.
                   The values in each column of matrix S should sum to 1.
    '''
    #different division method to handle division by 0
    S = np.divide(A, np.add.reduce(A), out=np.zeros_like(A), where=np.add.reduce(A) != 0)
    h = np.shape(S)[0]
    colSum = np.add.reduce(S)
    for i in range(0, h):
        if colSum[i] == 0:
            S[:, i] = 1/h
    return S



#--------------------------
def pagerank_v2(A):
    ''' 
        A simplified version of PageRank algorithm, which solves the sink node problem.
        Given an adjacency matrix A, compute the pagerank score of all the nodes in the network. 
        Input: 
                A: adjacency matrix, a numpy matrix of binary values. If there is a link from node i to node j, A[j][i] =1. Otherwise A[j][i]=0 if there is no link.
        Output: 
                x: the ranking scores, a numpy vector of float values, such as np.array([.3, .5, .2])
    '''

    # Initialize the score vector 
    num_nodes = A.shape[0]
    x_0 = np.ones(num_nodes)/num_nodes
    S = compute_S(A)
    x, n_steps = random_walk(S, x_0)

    return x
