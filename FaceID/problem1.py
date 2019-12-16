#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 1: 
    In this problem, you will get familiar with matrix eigen vectors and eigen values.

'''

#--------------------------
def Terms_and_Conditions():
    return True


#--------------------------
def compute_eigen_pairs(X):
    '''
        Compute the eigen vectors and eigen values of matrix X. 
        Input:
            X:  a numpy float matrix of shape p by p, Note, X should be a symmetric matrix.
        Output:
            Ep:  the eigen pairs of matrix X, a python list of length p. 
                Ep is a list as [(v1,e1), (v2, e2), ... ]
                Each element of Ep corresponds to one eigen pair (v,e), here v is an eigen value of matrix X, e is its eigen vector.
                Here v is a float scalar, and e is a numpy vector of length p.

        For example, suppose we have a 3x3 matrix:
            X =  1, 0, 0
                 0, 2, 0
                 0, 0, 3
        This matrix has three eigen pairs:
        v1= 1, e1 = [1,0,0]
        v2= 2, e2 = [0,1,0]
        v3= 3, e3 = [0,0,1]
        So in this example, the eigen pairs of matrix X should be
        Ep =[(1, [1,0,0]),
             (2, [0,1,0]), 
             (3, [0,0,1])]
        Here P[0] represents the first eigen pair (1,[1,0,0]), where the eigen value is 1, and the eigen vector is [1,0,0]

    '''
    v, e = np.linalg.eigh(X)
    Ep = []
    for vals in range(len(v)):
        Ep.append((v[vals], e[:, vals]))
    return Ep


#--------------------------
def sort_eigen_pairs(Ep, order = 'ascending'):
    '''
        Sort the eigen pairs in descending/ascending order of the eigen values. 
        Input:
            Ep:  the eigen pairs of matrix X, a python list of length p. 
                Ep is a list as [(v1,e1), (v2, e2), ... ]
                Each element of Ep corresponds to one eigen pair (v,e), here v is an eigen value of matrix C, e is its eigen vector.
                Here v is a float scalar, and e is a numpy vector of length p.
            order: a string of either 'ascending' or 'descending', whether to sort the eigen pairs in ascending or descending order of the eigen values.
        Output:
            Ep: the sorted list of eigen pairs, a python list of length p. 

        For example, suppose we have a 3x3 matrix:
            X =  2, 0, 0
                 0, 1, 0
                 0, 0, 3
        This matrix has three eigen pairs:
        v1= 2, e1 = [1,0,0]
        v2= 1, e2 = [0,1,0]
        v3= 3, e3 = [0,0,1]
        So in this example, the eigen pairs of matrix X should be
        Ep =[(2, [1,0,0]),
             (1, [0,1,0]), 
             (3, [0,0,1])]
        If we sort P in ascending order:
        Ep= [(1, [0,1,0]), 
             (2, [1,0,0]), 
             (3, [0,0,1])]
        If we sort P in descending order:
        Ep= [(3, [0,0,1]),
             (2, [1,0,0]), 
             (1, [0,1,0])]
    '''
    Ep.sort()
    if order == 'descending':
        Ep = Ep[::-1]
    return Ep 
