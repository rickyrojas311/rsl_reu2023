"""
Testing enviroment for anisotropic_operator_subclass
"""
from __future__ import annotations
import math

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import sigpy as sp
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import downsampling_subclass as spl
import projection_operator_subclass as proj
import anisotropic_class as anic

def is_transpose(input_op: sp.linop.Linop, ishape: tuple[int], oshape: tuple[int]):
    """
    Checks properties of the transpose of A to verify A.H is the transpose
    of A
    """
    x = xp.random.rand(*ishape)
    y = xp.random.rand(*oshape)
    A_x = input_op(x)
    A_T = input_op.H
    A_T_y = A_T(y)
    left = xp.vdot(A_x, y)
    right = xp.vdot(x, A_T_y)
    return left, right, math.isclose(left, right)

def find_mse(ground_truth: xp.ndarray, reconstruction: xp.ndarray):
    """
    Given two arrays of the same shape, finds the MSE between them
    """
    return xp.mean((ground_truth - reconstruction) ** 2)

def normalize_matrix(matrix):
    """
    normalizes inputed matrix so that all of it's values range from 0 - 1
    """
    m_max = matrix.max()
    return matrix/m_max

def produce_images():
    """
    Produces images
    """
    img_header = nib.as_closest_canonical(nib.load(r"C:\Users\ricky\OneDrive\Desktop\RSL REU\rsl_reu2023\project_data\BraTS_Data\BraTS_002\images\T1.nii"))
    ground_truth = img_header.get_fdata()[:, :, 100]
    img2_header = nib.as_closest_canonical(nib.load(r"C:\Users\ricky\OneDrive\Desktop\RSL REU\rsl_reu2023\project_data\BraTS_Data\BraTS_002\images\T2.nii"))
    structural_data = img2_header.get_fdata()[:, :, 100]
    A = spl.AverageDownsampling(ground_truth.shape, (8, 8))
    y = A(ground_truth)
    G = sp.linop.FiniteDifference(ground_truth.shape)
    P = proj.ProjectionOperator(G.oshape, structural_data)
    op = sp.linop.Compose([P,G])

    lambdas = [0, 1, 2, 3, 4, 16]

    recons = xp.zeros((len(lambdas),) + tuple(G.ishape))
    for i, lam in enumerate(lambdas):
        gproxy = sp.prox.L1Reg(op.oshape, lam)
        alg = sp.app.LinearLeastSquares(A, y, proxg=gproxy, G=op, max_iter=6000)
        recons[i,...] = alg.run()


    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,15))
    for i, recon in enumerate(recons):
        ax.ravel()[i].imshow(recon, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
        ax.ravel()[i].set_title("lam = " + str(lambdas[i]))
        ax.ravel()[i].axis("off")
    fig.show()

    """
    Sweeps the lamda value of linearleastsquares to find best result.

    min_lambda is the mininum lambda sweeped
    starting_interval is the interval sweeped every iteration, best if 2^n.
    max_iterations is the number of iterations ran by Linear Least Squares
    percision is the acceptable distance from the optimal lambda, 0 means the exact lambda.
    """
    #Sets up operator for image reconstruction
    A = spl.AverageDownsampling(ground_truth.shape, (8, 8))
    y = A(ground_truth)
    G = sp.linop.FiniteDifference(ground_truth.shape)
    P = proj.ProjectionOperator(G.oshape, structural_data, eta=eta)
    op = sp.linop.Compose([P,G])

    #Initalizes kwargs and first MSE
    lam = low_lam = min_lambda
    interval = starting_interval
    gproxy = sp.prox.L1Reg(op.oshape, lam)
    alg = sp.app.LinearLeastSquares(A, y, proxg=gproxy, G=op, max_iter=max_iterations)
    prev = curr = low = find_mse(ground_truth, alg.run())
    max_search = True
    lambdas = []
    lambdas.append((lam, interval, prev))
    lam += interval


    while interval > percision:
        #Finds MSE for current lam
        gproxy = sp.prox.L1Reg(op.oshape, lam)
        alg = sp.app.LinearLeastSquares(A, y, proxg=gproxy, G=op, max_iter=max_iterations)
        x = alg.run()
        next = find_mse(ground_truth, x)
        if next < low:
            low = next
            low_lam = lam
        lambdas.append((lam, interval, max_search, next))
        #Only runs when upperbound not yet determined
        if max_search and next < curr:
            lam += interval
            prev = curr
            curr = next
        #Runs once to start search of log(N)
        elif max_search:
            interval //= 2
            if prev < next and (lam - interval * 3) > 0:
                lam -= interval * 3
            else:
                lam -= interval
                prev = curr
                curr = next
            max_search = False
        #Searches for min MSE in log(N)
        else:
            interval //= 2
            if prev < curr:
                lam -= interval
                curr = next
            else: 
                lam += interval
                prev = next
    return (low_lam, low, x, lambdas)

def sweep_lambda_helper(ground_truth, lam, interval, percision, direction, oper: anic.AnatomicReconstructor, prev: float, low: tuple[float]) -> tuple[float, xp.array]:
    """
    Recursive helper for the sweep lambda function
    """
    if interval <= percision:
        oper.given_lambda = lam
        return low[0]
    else:
        interval /= 2
        oper.given_lambda = lam + direction * interval
        recon = oper(ground_truth)
        curr = find_mse(ground_truth, recon)
        print(curr, oper.given_lambda, interval)
        if curr < low[1]:
            low = (oper.given_lambda, curr)
        if curr < prev:
            return sweep_lambda_helper(ground_truth, oper.given_lambda, interval, percision, direction, oper, curr, low)
        else:
            oper.given_lambda = lam - direction * interval
            recon = oper(ground_truth)
            temp = find_mse(ground_truth, recon)
            if temp < low[1]:
                low = (oper.given_lambda, temp)
            if temp < curr:
                if temp < prev:
                    print(temp, oper.given_lambda, interval)
                    return sweep_lambda_helper(ground_truth, oper.given_lambda, interval, percision, -direction, oper, temp, low)
                else:
                    print(temp, oper.given_lambda, interval)
                    return sweep_lambda_helper(ground_truth, oper.given_lambda, interval, percision, direction, oper, temp, low)
            else:
                print(curr, lam + direction * interval, interval)
                return sweep_lambda_helper(ground_truth, lam + direction * interval, interval, percision, direction, oper, curr, low)

def sweep_lambda(ground_truth, oper: anic.AnatomicReconstructor, min_lambda: int = 0, starting_interval: int = 1e-2, percision: int = 1e-4):
    """
    Function to sweep lambda values until the lambda that minimizes MSE values is found.
    Starts by sweeping to find the upperbound then calls the helper function to search the inner bounds
    """
    if oper.normalize:
        ground_truth = normalize_matrix(ground_truth)
    lam = min_lambda - starting_interval
    prev = None
    curr = None
    while prev is None or prev > curr:
        prev = curr
        lam += starting_interval
        oper.given_lambda = lam
        recon = oper(ground_truth)
        curr = find_mse(ground_truth, recon)
        print(curr, lam)
    result = sweep_lambda_helper(ground_truth, lam - starting_interval, starting_interval, percision, 1, oper, prev, (lam, curr))
    return result

def sweep_eta_helper(ground_truth, eta, interval, percision, direction, oper: anic.AnatomicReconstructor, prev: float, low: tuple[float]) -> tuple[float, xp.array]:
    """
    Recursive helper for the sweep eta function
    """
    if interval <= percision:
        oper.given_eta = eta
        return low[0]
    else:
        interval /= 2
        oper.given_eta = eta + direction * interval
        recon = oper(ground_truth)
        curr = find_mse(ground_truth, recon)
        print(curr, oper.given_eta, interval)
        if curr < low[1]:
            low = (oper.given_eta, curr)
        if curr < prev:
            return sweep_eta_helper(ground_truth, oper.given_eta, interval, percision, direction, oper, curr, low)
        else:
            oper.given_eta = eta - direction * interval
            recon = oper(ground_truth)
            temp = find_mse(ground_truth, recon)
            if temp < low[1]:
                low = (oper.given_eta, temp)
            if temp < curr:
                if temp < prev:
                    print(temp, oper.given_eta, interval)
                    return sweep_eta_helper(ground_truth, oper.given_eta, interval, percision, -direction, oper, temp, low)
                else:
                    print(temp, oper.given_eta, interval)
                    return sweep_eta_helper(ground_truth, oper.given_eta, interval, percision, direction, oper, temp, low)
            else:
                print(curr, eta + direction * interval, interval)
                return sweep_eta_helper(ground_truth, eta + direction * interval, interval, percision, direction, oper, curr, low)

def sweep_eta(ground_truth, oper: anic.AnatomicReconstructor, min_eta: int = 0, starting_interval: int = 1e-2, percision: int = 1e-4):
    """
    Function to sweep eta values until the eta that minimizes MSE values is found.
    Starts by sweeping to find the upperbound then calls the helper function to search the inner bounds
    """
    if oper.normalize:
        ground_truth = normalize_matrix(ground_truth)
    eta = min_eta - starting_interval
    prev = None
    curr = None
    while prev is None or prev > curr:
        prev = curr
        eta += starting_interval
        if eta == 0:
            oper.given_eta = 1e-9
        else:
            oper.given_eta = eta
        recon = oper(ground_truth)
        curr = find_mse(ground_truth, recon)
        print(curr, eta)
    result = sweep_eta_helper(ground_truth, eta - starting_interval, starting_interval, percision, 1, oper, prev, (eta, curr))
    return result

def check_lambda_on_mse(ground_truth, structural_data, lambda_min, intervals, lambda_num, max_iterations, saving_options):
    """
    Plots MSE score by lamda value
    """
    lam = lambda_min
    op = anic.AnatomicReconstructor(_structural_data, (8,8), lam, 0.0016, 1000, True, saving_options)
    x = op(_ground_truth)
    lambdas = []
    MSEs = []

    for i in range(lambda_num):
        lam = i * intervals
        lambdas.append(lam)
        gproxy = sp.prox.L1Reg(op.oshape, lam)
        alg = sp.app.LinearLeastSquares(A, y, proxg=gproxy, G=op, max_iter=max_iterations)
        MSE = find_mse(ground_truth, alg.run())
        MSEs.append(MSE)
    
    x_axis = lambdas
    y_axis = MSEs

    plt.plot(x_axis, y_axis)
    plt.title('Lambda vs MSE')
    plt.xlabel('lambdas')
    plt.ylabel('MSE')
    plt.show()

    return (lambdas, MSEs)

def diff_images(ground_truth, structural_data, saving_options):
    """
    Shows the difference between the ground truth and the reconstruction
    """
    op = anic.AnatomicReconstructor(structural_data, (8,8), 2, 0.0015, 6000, True, saving_options)
    ground_truth = normalize_matrix(ground_truth)
    lambdas = [1e-4,1e-3,5e-3,1e-2,1e-1,1]

    recons = xp.zeros((len(lambdas),) + ground_truth.shape)
    for i, lam in enumerate(lambdas):
        op.given_lambda = lam
        image = op(ground_truth) - ground_truth
        # import ipdb; ipdb.set_trace()
        recons[i,...] = image

    vmax = xp.abs(recons).max()
    # ipdb.set_trace()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,15))
    for i, recon in enumerate(recons):
        ax.ravel()[i].imshow(recon, vmin = -vmax, vmax = vmax, cmap = 'seismic')
        ax.ravel()[i].set_title("lam = " + str(lambdas[i]))
        ax.ravel()[i].axis("off")
    fig.show()
    fig.savefig(r"C:\Users\ricky\OneDrive\Desktop\RSL REU\rsl_reu2023\project_data\BraTS_Reconstructions\Composites\difference_masked.png")


if __name__ == "__main__":
    _img1_header = nib.as_closest_canonical(nib.load(r"project_data\BraTS_Data\Noise_Experiments\DMI_patient_9_ds_11_gm_4.0_wm_1.0_tumor_6.0_noise_0.001\dmi_gt.nii.gz"))
    _ground_truth = _img1_header.get_fdata()[:, :, 56, 0]
    _ground_truth = normalize_matrix(_ground_truth)
    _img2_header = nib.as_closest_canonical(nib.load(r"project_data\BraTS_Data\Noise_Experiments\DMI_patient_9_ds_11_gm_4.0_wm_1.0_tumor_6.0_noise_0.001\dmi.nii.gz"))
    _low_res_data = _img2_header.get_fdata()[:, :, 56, 0]
    _low_res_data = normalize_matrix(_low_res_data)
    _img3_header = nib.as_closest_canonical(nib.load(r"project_data\BraTS_Data\BraTS_009\images\FLAIR.nii"))
    _structural_data = _img3_header.get_fdata()[:, :, 79]
    _structural_data = normalize_matrix(_structural_data)

    save_options = {"given_path": r"project_data\BraTS_Noise_Expirements_Reconstructions", "img_header": _img1_header}
    _op = anic.AnatomicReconstructor(_structural_data, _low_res_data, (11, 11), 0.007, 0.0015, 8000, True, save_options)
    recon = _op(_ground_truth)

    plt.imshow(recon, "Grey_r", vmin=0, vmax=_ground_truth)
    plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,15))
    # ax.ravel()[0].imshow(_ground_truth, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[0].set_title("ground truth")
    # ax.ravel()[0].axis("off")
    # ax.ravel()[1].imshow(_low_res_data, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[1].set_title("low res")
    # ax.ravel()[1].axis("off")
    # ax.ravel()[2].imshow(_structural_data, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[2].set_title("structure")
    # ax.ravel()[2].axis("off")
    # fig.show()
    # fig.savefig(r"project_data\BraTS_Noise_Expirements_Reconstructions\patient_9_all_data_side_by")


