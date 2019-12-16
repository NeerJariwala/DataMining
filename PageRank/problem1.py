#-------------------------------------------------------------------------
'''
    Problem 1: getting familiar with python and unit tests.
    In this problem, please install python version 3 and the following package:
        * nose   (for unit tests)

    To install python packages, you can use any python package management software, such as pip, conda. For example, in pip, you could type `pip3 install nose` in the terminal to install the package.

    Then start implementing function swap().
    '''

#--------------------------
def Terms_and_Conditions():
    return True


 

#--------------------------
def swap( A, i, j ):
    ''' 
        Swap the i-th element and j-th element in list A.  
        Inputs: 
            A:  a list, such as  [2,6,1,4]
            i:  an index integer for list A, such as  3
            j:  an index integer for list A, such as  0
    '''
    x = A[i]
    A[i] = A[j]
    A[j] = x


#--------------------------
def sort_list( A ):
    ''' 
        Given a disordered list of integers, rearrange the integers in natural order using bubble sort algorithm.
        Input: A:  a list, such as  [2,6,1,4]
        Output: A should be sorted, such as [1,2,4,6]
    '''
    length = len(A)
    # all elements
    for i in range(length):
        # dont worry about already sorted ones
        for j in range(0, length-i-1):
            if A[j] > A[j+1]:
                swap(A, j, j+1)
