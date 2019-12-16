#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_olivetti_faces
import problem2 as p2 
#-------------------------------------------------------------------------
'''
    Problem 3: eigen faces 
    In this problem, you will use PCA to compute eigen faces in a face image dataset.
    We will use olivetti face image dataset (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
    It include 400 face images, and 40 human subjects. For each human subject, we have 10 face images.
    Each face image is a 64 by 64 black-and-white image.
    You need to install the following package:
        * sklearn
        * imageio 
    You could use `pip3 install sklearn`                        
                  `pip3 install imageio` to install the packages.

    Notations:
            ---------- input data ------------------------
            n: the number of face images, an integer scalar. (n = 400)
            p: the number of pixels in each image, an integer scalar. (p= 4096)
            X: the feature matrix, a float numpy matrix of shape n by p. (400 by 4096)
            y: labels associated to each face image, a numpy integer vector of length n. (n=400)
               The i-th element can be  0,1,..., or 39 corresponding to the Subject ID of the i-th image. 
            mu: the average vector of matrix X, a numpy float vector of length  p. (p=4096)
                Each element mu[i] represents the average value in the i-th column of matrix X.
            mu_image: the average face image, a numpy float matrix of shape h by w (64 by 64). 
                      Each element mu[i,j] represents the average value in the ij-th pixel in all face images. 
            k: the number of dimensions to reduce to, an integer scalar.
            P_images:  the eigen faces, a python list of length k. 
                Each element in the list is an eigen face image, which is a numpy float matrix of shape 64 by 64. 
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape n by k. 
'''

#--------------------------
def vector_to_image(x):
    '''
        Reshape a feature vector into a face image 
        Input:
            x:  a feature vector , a float numpy vector of length k*k. 
        Output:
            image: the face image, a float numpy matrix of shape (k,k). 
        For example, suppose we have a vector [1,2,3,4], after reshaping (k=2), the image matrix should be 
            [[1,2],
             [3,4]]
    '''
    k = int(np.sqrt(len(x)))
    image = np.reshape(x, (k, k))
    return image


#--------------------------
def load_dataset():
    '''
        Load (or download if not exist) the olivetti face image dataset (http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html).
        Output:
            X: the feature matrix, a float numpy matrix of shape n by p. (400 by 4096)
               The i-th row of X represent the i-th image (a 64 by 64 image, reshaped into a 4096 dimensional vector) in the dataset.
            y: labels associated to each face image, a numpy integer vector of length n (n=400 in this dataset).
               The i-th element can be  0,1,..., or 39 corresponding to the Subject ID of the i-th image. 
            images: numpy array of shape (400, 64, 64). Each face image is a (64, 64) matrix, and we have 400 images in the dataset.
    '''
    data = fetch_olivetti_faces()
    X = data.data
    y = data.target
    images = data.images
    return X, y, images



#--------------------------
def compute_mu_image(X):
    '''
        Compute the average face image in the dataset . 
        Input:
            X:  the feature matrix, a float numpy matrix of shape (400, 4096). Here 400 is the number of images, 4096 is the number of features.
        Output:
            mu_image:  the average face image, a float numpy matrix of shape (64,64). Hint: you could reshape a vector of length 4096 into a matrix of shape 64 X 64
        Hint: you need first compute the average vector of matrix X. 
                The length of the average vector is (4096,), which you can reshape into a matrix of shape 64 by 64.
    '''
    temp = X.mean(axis=0)
    mu_image = vector_to_image(temp)
    return mu_image


#--------------------------
def compute_eigen_faces(X, k=20):
    '''
        Compute top k eigen faces of the olivetti face image dataset using PCA.
        Each eigen face corresponds to an eigen vector in the PCA on matrix X.
        For example, suppose the top k eigen vectors in the PCA on matrix X are e1, e2, e3  (suppose k=3),
        Then e1 is a vector of length 4096, if we reshape e1, we can get the first eigen face image is a matrix of shape 64x64.
        Similar, reshaping e2, we get the second eigen face image.
        Input:
            X:  the feature matrix, a float numpy matrix of shape (400, 4096). Here 400 is the number of images, 4096 is the number of features.
            k:  the number of eigen face to keep. 
        Output:
            P_images:  the eigen faces, a python list of length k. 
                The i-th element in the list is the i-th eigen face image, which is a numpy float matrix of shape 64 by 64. 
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape n by k. (400 by k) 
        Note: this function may take 1-5 minutes and 1-2GB of memory to run.
    '''
    Xp, P = p2.PCA(X, k)
    P_images = []
    for i in range(k):
        P_images.append(vector_to_image(P[:, i]))
    return P_images, Xp
