#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
import problem1 as p1
#-------------------------------------------------------------------------
'''
    Problem 2: PCA 
    In this problem, you will implement a version of the principal component analysis method to reduce the dimensionality of data.

    Notations:
            ---------- input data ------------------------
            n: the number of data instances (for example, # of images), an integer scalar.
            p: the number of dimensions (for example, # of pixels in each image), an integer scalar.
            X: the feature matrix, a float numpy matrix of shape n by p. 
            ---------- computed data ----------------------
            mu: the average vector of matrix X, a numpy float matrix of shape 1 by p. 
                Each element mu[0,i] represents the average value in the i-th column of matrix X.
            Xc: the centered matrix X, a numpy float matrix of shape n by p. 
                Each column has a zero mean value.
            C:  the covariance matrix of matrix X, a numpy float matrix of shape p by p. 
            k:  the number of dimensions to reduce to (k should be smaller than p), an integer scalar
            Xp: the projected feature matrix with reduced dimensions, a numpy float matrix of shape n by k. 
             P: the projection matrix, a numpy float matrix of shape p by k. 
            -----------------------------------------------
'''

#--------------------------
def centering_X(X):
    '''
        Centering matrix X, so that each column has zero mean.
        Input:
            X:  the feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
        Output:
            Xc:  the centered matrix X, a numpy float matrix of shape n by p. 
            mu:  the average row vector of matrix X, a numpy float vector of length p. 
        Note: please don't use the np.cov() function. There seems to be a bug in their code which will result in an error in later test cases. 
              Please implement this function only using basic numpy functions, such as np.mean().
    '''
    mu = X.mean(axis=0)
    Xc = X - mu
    return Xc, mu



#--------------------------
def compute_C(Xc):
    '''
        Compute the covariance matrix C from the data matrix Xc (centered). 
        Input:
            Xc:  the centered feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
        Output:
            C:  the covariance matrix, a numpy float matrix of shape p by p. 
        Note: please don't use the np.cov() function here. Implement the function using matrix multiplication.
    '''
    C = np.dot(Xc.T, Xc) / (np.size(Xc, 0) - 1)
    return C



#--------------------------
def compute_P(C,k):
    '''
        Compute the projection matrix P by combining the k eigen vectors, that correspond to the top k largest eigen values of matrix C. 
        Here the projection matrix P includes all the k principle components.
        Input:
            C:  the covariance matrix, a numpy float matrix of shape p by p. 
            k:  the number of dimensions to reduce to (k should be smaller than p), an integer scalar
        Output:
            P: the projection matrix, a numpy float matrix of shape p by k. 
                For example, if we sort the eigen pairs of matrix C in descending order, and 
                the result [(v1,e1),(v2,e2),...., (vp, ep)], here v1 >= v2 >= v3 ...
                The projection matrix should be [e1^T, e2^T, ..., ek^T], here e^T represents the transpose of a row vector e (which is a column vector)
    '''
    Ep = p1.compute_eigen_pairs(C)
    Ep = p1.sort_eigen_pairs(Ep, 'descending')
    height = np.size(C, 0)
    P = np.empty([height, k], dtype=float)
    for j in range(k):
        for i in range(height):
            P[i, j] = Ep[j][1][i]
    return P



#--------------------------
def compute_Xp(Xc,P):
    '''
        Compute the projected feature matrix Xp by projecting data Xc using matrix P. 
        Input:
            Xc:  the feature matrix after centering, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
             P: the projection matrix, a numpy float matrix of shape p by k. 
        Output:
            Xp: the feature matrix after projection (dimension reduced), a numpy float matrix of shape n by k. 
    '''
    Xp = np.dot(Xc, P)
    return Xp



#--------------------------
def PCA(X, k=1):
    '''
        Compute PCA of matrix X. 
        Input:
            X:  the feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
            k:  the number of dimensions to output (k should be smaller than p)
        Output:
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape n by k. 
             P: the projection matrix, a numpy float matrix of shape p by k. 
        Note: in this problem, you should not use existing packages for PCA, such as scikit-learn
    '''

    # centering matrix X
    Xc, mu = centering_X(X)


    # compute covariance matrix C
    C = compute_C(Xc)

    # compute the projection matrix P 
    P = compute_P(C, k)

    # project the data into lower dimension using projection matrix P and centered data matrix X
    Xp = compute_Xp(Xc, P)


    return Xp, P 
