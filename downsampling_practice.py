"""
Practice using the sigpy library to create a downsampling operator and its transpose 
as well as verfiy that it is actually A^T
"""

import math
import numpy as np
import sigpy as sp

import nibabel as nib
import matplotlib.pyplot as plt

import downsampling_subclass as spl


def is_tranpose(ishape, factors):
    """
    Checks properties of the transpose of A to verify A.H is the transpose
    of A
    """
    vectorA_size = np.prod(ishape)
    A = spl.AverageDownsampling((vectorA_size,), factors)
    A_transpose = A.H
    x = np.random.rand(*ishape).flatten()
    A_x = A * x
    y = np.random.rand(A_x.size)
    left = np.dot(A_x, y)
    right = np.dot(x, A_transpose * y)
    return left, right, math.isclose(left, right)

def size_test(shape, factor):
    """
    Test case for Downsampling/Upsampling subclasses based on random matrices
    """
    
    print("Starting Matrix: \n", x := np.random.rand(*shape))
    print(factor)
    A = spl.AverageDownsampling(shape, factor)
    print("Downsample: \n", y := A(x))
    print("Upsample: \n", x_prime := A.H * y)
    print("Is tranpose:", is_tranpose(shape, factor))

if __name__ == "__main__":
    np.random.seed(100)
    # size_test((240, 240, 155), (8, 8, 5))

    img_header = nib.as_closest_canonical(nib.load(r"C:\Users\ricky\OneDrive\Desktop\RSL REU\rsl_reu2023\project_data\BraTS_Data\BraTS_002\images\T1.nii"))
    ground_truth = img_header.get_fdata()[:, :, 100]
    A = spl.AverageDownsampling(ground_truth.shape, (8, 8))
    y = A(ground_truth)
    x0 = np.random.rand(*A.ishape)
    alg = sp.app.LinearLeastSquares(A, y, x = x0)
    x = alg.run()

    x2 = A.H(y)

    fig, ax = plt.subplots(nrows=1, ncols=4 ,figsize=(20,15))
    ax[0].imshow(ground_truth, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax[1].imshow(y, vmin = 0, vmax = y.max(), cmap = 'Greys_r')
    ax[2].imshow(x, vmin = 0, vmax = x.max(), cmap = 'Greys_r')
    ax[3].imshow(x2, vmin = 0, vmax = x2.max(), cmap = 'Greys_r')
    fig.show()
