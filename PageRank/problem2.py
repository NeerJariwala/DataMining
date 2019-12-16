import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 2: getting familiar with numpy package.
    In this problem, please install the following python package:
        * numpy 
    Numpy is the library for scientific computing in Python. 
    It provides a high-performance multidimensional array object, and tools for working with these arrays. 
    To install numpy using pip, you could type `pip3 install numpy` in the terminal.
'''


#--------------------------
def ndarray():
    ''' 
       Create the following 2 x 3 matrix using nd-array in NumPy:
            1,2,3
            4,5,6 
        Output: 
                X: a numpy matrix of shape 2 X 3, the matrix, each element is an integer
    '''
    X = np.array([[1, 2, 3], [4, 5, 6]])
    return X



#--------------------------
def float_array():
    ''' 
       Create the following 2 X 3 matrix using nd-array in NumPy:
            0.1, 0.2, 0.3
            0.4, 0.5, 0.6 
        Output: 
                X: a numpy matrix of shape 2 X 3, the matrix, each element is a float number
    '''
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float64)
    return X



#--------------------------
def get_shape(X):
    ''' 
        Given a NumPy matrix, return the number of rows and columns of the matrix
        Input: 
                X: a numpy matrix 
        Output: 
                h: an integer, the hight of the matrix x (number of rows)
                w: an integer, the width of the matrix x (number of columns)
    '''
    h = np.shape(X)[0]
    w = np.shape(X)[1]
    return h, w 


#--------------------------
def all_one_matrix(m,n):
    ''' 
        Create a numpy matrix of shape m X n, all the values in the matrix should be 1.0 
        Input: 
                m: an integer scalar, the number of rows in the matrix
                n: an integer scalar, the number of columns in the matrix
        Output: 
                X: a numpy matrix of shape m X n, the matrix, each element is a float number of value 1.0
    '''
    X = np.ones((m,n))
    return X 



#--------------------------
def mat_sum(X):
    ''' 
        Given a matrix X of shape m x n, compute the sum of each column in the matrix 
        Input: 
                X: a numpy matrix of shape m X n 
        Output: 
                s: a numpy vector of shape (n,) the i-th element of s is the sum of the i-th column of matrix X 
    '''
    s = np.add.reduce(X)
    return s



#--------------------------
def mat_scalar_multiplication(X, c):
    ''' 
        Given a matrix X of shape m x n, and a scalar c, the compute the product between the matrix and scalar: Y = cX
        For example, if matrix X is: 
                                1,2
                                3,4
        and c = 2

        Then Y = cX should be:
                                2,4
                                6,8

        Input: 
                X: a numpy matrix of shape m X n 
                c: a float scalar 
        Output: 
                Y: a numpy matrix of shape (m,n),  each element Y[i,j] = c*X[i,j] 
    '''
    Y = X*c
    return Y


#--------------------------
def matrix_vector_multiplication(X, y):
    ''' 
        Given a matrix X and a vector y, compute the product X*y = z
        For example, if matrix X is: 
                                1,2
                                3,4
        and vector y is:
                                5
                                10

        Then z = X y should be:
                                25      =  5*1 + 10 * 2
                                55      =  5*3 + 10 * 4

        Input: 
                X: a numpy matrix of shape m x n, 
                y: a numpy vector of shape n x 1, 
        Output: 
                z: the numpy vector of shape m x 1, the result of the matrix vector product. 
    '''
    z = X.dot(y)
    return z



#--------------------------
def find_zeros(x):
    ''' 
        Given a vector x of length  n, find indices of all zeros elements in x.
        For example, if vector x is: 
                                1,0,4,0
        Then d should be:
                                1,3

        Input: 
                x: a numpy vector of length n 
        Output: 
                d: a numpy vector of length m (m is the number zeros in vector x), the i-th element of d is the index of the i-th zero in x. 
    '''
    d = np.where(x == 0)[0]
    return d



#--------------------------
def diag_mat(x):
    ''' 
        Given a vector x of length  n, create a diagonal matrix D where the i-th diagonal element D[i,i] = x[i]
        All the non-diagonal elements of D are zeros: D[i,j] = 0, if i not equal to j.
        For example, if vector x is: 
                                1,2,3
        Then D should be:
                                1,0,0
                                0,2,0
                                0,0,3

        Input: 
                x: a numpy vector of length n 
        Output: 
                D: a numpy of shape n x n 
    '''
    D = np.diag(x)
    return D
