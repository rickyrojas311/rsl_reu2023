"""
Practice Finding the argmin of X using gradient descent
"""
import numpy as np

def get_delta_f(matrix_a, vector_x, vector_d):
    """
    calculates the delta of function f
    """
    a_transpose = np.transpose(matrix_a)
    return a_transpose @ (matrix_a @ vector_x - vector_d)
#test


def min_gradient_descent(matrix_a, vector_d, stepsize=0.1, max_iterations=100):
    """
    Uses gradient descent to calculate the local minimum of a function
    Stepsize controls the distance traveled by the algorithm towards the minimum
    max_iterations caps the number of steps the algoritm preforms
    """
    #Starts from 0
    vector_x = np.zeros(matrix_a.shape[1])
    delta_f = get_delta_f(matrix_a, vector_x, vector_d)
    num_iterations = 0
    while num_iterations < max_iterations:
        vector_x -= stepsize * delta_f
        delta_f = get_delta_f(matrix_a, vector_x, vector_d)
        num_iterations += 1
        print(vector_x, delta_f)
    return vector_x


def test_regular_matrix():
    """
    tests mine_gradient_descent on a basic matrix
    """
    A = np.array([[1,0],[3,2],[4,5]])
    d = np.array([1,0,3])
    return min_gradient_descent(A, d)

def test_simple_matrix():
    """
    tests min_gradient_descent on a simple matrix
    """
    A = np.array([[2,1],[4,1]])
    d = np.array([1,0])
    return min_gradient_descent(A, d)

if __name__ == '__main__':
    np.random.seed(1)

    A = np.random.rand(3,2)
    d = np.random.rand(3)

    #A = np.array([[1,0],[3,2],[4,5]])
    #d = np.array([1,0,3])

    min_gradient_descent(A, d, stepsize=1., max_iterations=10000)