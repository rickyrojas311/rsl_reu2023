"""
toy inverse problem to find and display a graph of 
"""
import matplotlib.pyplot as plt

def get_delta(x0, y):
    """
    Calculates Delta by expanding (x0 + deltaX)^2 = y and assuming deltaX ^2 ~= 0
    """
    return (y - x0 ** 2)/(2 * x0)


def create_convergence_plot(y, iterations, convergence):
    """
    creates a plot of how quickly the solution to sqrt(y) is coverging on the actual answer
    """
    x_axis = iterations
    y_axis = convergence

    plt.plot(x_axis, y_axis)
    plt.title(f'Convergence to the solution of sqrt({y})')
    plt.xlabel('Iterations')
    plt.ylabel('Convergence')
    plt.show()


def sqrt(y, percision=15, plot_convergence=False):
    """
    Calculates the sqrt of y through continually calling getDelta until x0 converges
    to the sqrt of y or reaches the max numver of digits

    kwarg** percision sets the number of interations sqrt will run for
    """
    x0 = y/2
    delta_x = get_delta(x0, y)
    num_iterate = 0

    if plot_convergence:
        convergence = []
        iterations = []
    while x0 ** 2 != y and num_iterate <= percision:
        x0 += delta_x
        delta_x = get_delta(x0, y)
        num_iterate += 1
        if plot_convergence:
            convergence.append(x0 ** 2 - y)
            iterations.append(num_iterate)
    if plot_convergence:
        create_convergence_plot(y, iterations, convergence)
    return x0


def test_prefect_sqrt():
    """
    Tests perfect squares
    """
    assert sqrt(1) == 1
    assert sqrt(4) == 2
    assert sqrt(16) == 4

def test_imperfect_sqrt():
    """
    Tests numbers that don't perfectly resolve into integers
    """
    assert sqrt(3, 15, True) == 1.7320508075688772
    assert sqrt(6.25) == 2.5
    # assert sqrt(2, 4) == 1.4142
