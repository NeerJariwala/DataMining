import numpy as np
from problem5 import pagerank 

#-------------------------------------------------------------------------
'''
    Problem 6: use PageRank (implemented in Problem 5) to compute the ranked list of nodes in a real-world network. 
    In this problem, we import a real-world network and use pagerank algorithm to rank the nodes in the network.
    File `network.csv` contains a network adjacency matrix. 
    (1) import the network from the file
    (2) compute the pagerank scores for the network
'''

#--------------------------
def import_A(filename ='network.csv'):
    '''
        import the addjacency matrix A from a CSV file
        Input:
                filename: the name of csv file, a string 
        Output: 
                A: the ajacency matrix, a numpy matrix of shape (n by n)
    '''
    A = np.loadtxt(filename, delimiter=",")
    return A


#--------------------------
def score2rank(x):
    '''
        compute a list of node IDs sorted by descending order of pagerank scores in x.
        Note the node IDs start from 0. So the IDs of the nodes are 0,1,2,3, ...
        
        For example, suppose we have 3 nodes, and their pagerank scores are
                0.2
            x=  0.1
                0.3

        Then the sorted ID of these three pages should be

            sorted_ids = [2,0,1]
        Because the node with the largest score (x[2]=0.3) is the last node (index = 2);
                the second largest node (x[0]=0.2) has an index = 0 (the first node)
                the smallest node (x[1]=0.1) has an index = 1 (the first node)

        Input: 
                x: the numpy vector of pagerank scores, length n
        Output: 
                sorted_ids: a numpy array of node IDs (starting from 0) in descending order of their pagerank scores, a python list of integer values, such as [2,0,1,3].
    '''
    n = np.size(x)
    sorted_ids = np.argsort(-x)[:n]
    return sorted_ids


#--------------------------
def node_ranking(filename = 'network.csv', alpha = 0.95):
    ''' 
        Rank the nodes in the network imported from a CSV file.
        (1) import the adjacency matrix from `filename` file.
        (2) compute pagerank scores of all the nodes
        (3) return a list of node IDs sorted by descending order of pagerank scores 

        Input: 
                filename: the csv filename for the adjacency matrix, a string.
                alpha: a float scalar value, which is the probability of choosing option 1 (randomly follow a link on the node)

        Output: 
                sorted_ids: the list of node IDs (starting from 0) in descending order of their pagerank scores, a python list of integer values, such as [2,0,1,3].
    '''
    A = import_A(filename)
    x = pagerank(A, alpha)
    sorted_ids = score2rank(x)
    return sorted_ids
