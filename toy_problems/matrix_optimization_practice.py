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


def optimal_stepsize(matrix_a):
    """
    Uses the lipshitz constant to find the optimal stepsize for matrix A
    """
    # return 1/np.linalg.norm(matrix_a)
    return 1/power_iteration(matrix_a, 1000)


def power_iteration(matrix_A, num_iterations: int):
    """
    Uses power_iteration formula to find the norm of Matrix A
    """
    A = matrix_A.T @ matrix_A
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = A @ b_k

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k1_norm


def min_gradient_descent(matrix_a, vector_d, stepsize=0, max_iterations=1000):
    """
    Uses gradient descent to calculate the local minimum of a function
    Stepsize controls the distance traveled by the algorithm towards the minimum
    max_iterations caps the number of steps the algoritm preforms
    """
    #Starts from 0
    vector_x = np.zeros(matrix_a.shape[1])
    old_x = np.ones(matrix_a.shape[1])
    if stepsize == 0:
        stepsize = optimal_stepsize(matrix_a)
    num_iterations = 0
    while num_iterations < max_iterations and not np.allclose(vector_x, old_x, rtol=1e-10, atol=1e-10):
        old_x = vector_x.copy()
        delta_f = get_delta_f(matrix_a, vector_x, vector_d)
        vector_x -= stepsize * delta_f
        num_iterations += 1
        print(vector_x, old_x, delta_f, stepsize, num_iterations)
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

    # A = np.random.rand(3,2)
    # d = np.random.rand(3)

    A = np.array([[1,0],[3,2],[4,5]])
    d = np.array([1,0,3])

    min_gradient_descent(A, d, max_iterations=1000)