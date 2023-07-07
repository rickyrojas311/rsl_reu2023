"""
Testing enviroment for anisotropic_operator_subclass
"""
import math

import numpy as np
import sigpy as sp
import nibabel as nib
import matplotlib.pyplot as plt

import downsampling_subclass as spl
import projection_operator_subclass as proj

def is_transpose(op, ishape, oshape):
    """
    Checks properties of the transpose of A to verify A.H is the transpose
    of A
    """
    x = np.random.rand(*ishape)
    y = np.random.rand(*oshape)
    A_x = op(x)
    A_T = op.H
    A_T_y = A_T(y)
    left = np.vdot(A_x, y)
    right = np.vdot(x, A_T_y)
    return left, right, math.isclose(left, right)

if __name__ == "__main__":
    img_header = nib.as_closest_canonical(nib.load(r"C:\Users\ricky\OneDrive\Desktop\RSL REU\rsl_reu2023\project_data\BraTS_Data\BraTS_002\images\T1.nii"))
    ground_truth = img_header.get_fdata()[:, :, 100]
    img2_header = nib.as_closest_canonical(nib.load(r"C:\Users\ricky\OneDrive\Desktop\RSL REU\rsl_reu2023\project_data\BraTS_Data\BraTS_002\images\T2.nii"))
    structural_data = img2_header.get_fdata()[:, :, 100]
    A = spl.AverageDownsampling(ground_truth.shape, (8, 8))
    y = A(ground_truth)
    G = sp.linop.FiniteDifference(ground_truth.shape)
    P = proj.ProjectionOperator(G.oshape, structural_data)
    # op = sp.linop.Compose([P,G])
    # gproxy = sp.prox.L1Reg(op.oshape, 35)
    # alg = sp.app.LinearLeastSquares(A, y, proxg=gproxy, G=op, max_iter=10000)
    # x = alg.run()

    lambdas = [10, 30, 50, 70, 90]
    etas = [1e3, 1e1, 1e-1, 1e-3, 1e-5]

    recons = np.zeros((len(lambdas) * len(etas),) + tuple(G.ishape))
    for i, lam in enumerate(lambdas):
        for j, eta in enumerate(etas):
            print(i, lam, j, eta)
            P.eta = eta
            op = sp.linop.Compose([P,G])
            gproxy = sp.prox.L1Reg(op.oshape, lam)
            alg = sp.app.LinearLeastSquares(A, y, proxg=gproxy, G=op, max_iter=2000)
            recons[i,...] = alg.run()
        
    
    # fig, ax = plt.subplots(nrows=2, ncols=2 ,figsize=(10,10))
    # ax.ravel()[0].imshow(ground_truth, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[0].set_title("Ground Truth")
    # ax.ravel()[0].axis("off")
    # ax.ravel()[1].imshow(y, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[1].set_title("Low Res")
    # ax.ravel()[1].axis("off")
    # ax.ravel()[2].imshow(x, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[2].set_title("Reconstructed")
    # ax.ravel()[2].axis("off")
    # ax.ravel()[3].imshow(structural_data, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[3].set_title("Structual Data")
    # ax.ravel()[3].axis("off")


    # fig.tight_layout()
    # fig.show()
    # # fig.savefig("4by4.png")

    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(20,15))
    title_dict = {'fontsize': 10}
    for i, recon in enumerate(recons):
        ax.ravel()[i].imshow(recon, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
        ax.ravel()[i].set_title("lam = " + str(lambdas[i // 5]) + " eta = " + str(etas[i % 5]), fontdict=title_dict, pad=-0.4)
        ax.ravel()[i].axis("off")
    fig.show()
    fig.savefig("lambda_vs_eta.png")

    