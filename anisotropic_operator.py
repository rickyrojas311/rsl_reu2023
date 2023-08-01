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

import downsampling_subclass as spl
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

def diff_images(ground_truth, oper: anic.AnatomicReconstructor, variables, variable):
    """
    Shows the difference between the ground truth and the reconstruction
    """
    recons = xp.zeros((len(variables),) + ground_truth.shape)
    for i, var in enumerate(variables):
        if variable == "lambda":
            oper.given_lambda = var
        elif variable == "eta":
            oper.given_eta = var
        else:
            raise ValueError(f"Invalid type {variable}")
        image = oper(oper.low_res_data)
        recons[i,...] = image

    vmax = xp.abs(recons).max()

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20,15))
    for i, recon in enumerate(recons):
        diff_recon = recon - ground_truth
        ax[0][i].imshow(diff_recon.get(), vmin = -vmax, vmax = vmax, cmap = 'seismic')
        mse=float(find_mse(ground_truth, diff_recon))
        ax[0][i].set_title(f"{variable}={variables[i]},mse={round(mse, 5)}")
        ax[0][i].axis("off")
        ax[1][i].imshow(recon.get(), vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
        ax[1][i].axis("off")
    fig.show()
    fig.savefig(f"project_data/BraTS_Reconstructions/Composites/differences_{variables[0]}to{variables[-1]}.png")

def compare_aquisitions():
    """
    Compare which scan produces the best reconstruction
    """
    ground_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/Noise_Experiments/DMI_patient_9_ds_11_gm_4.0_wm_1.0_tumor_6.0_noise_0.001/dmi_gt.nii.gz"))
    ground_truth = ground_header.get_fdata()[:, :, 56, 0]
    ground_truth = xp.array(normalize_matrix(ground_truth))
    T2_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t2.nii.gz"))
    structural_data = T2_header.get_fdata()[:, :, 56]
    structural_data = normalize_matrix(structural_data)

    down = spl.AverageDownsampling(ground_truth.shape, (11, 11))
    low_res_data = down(ground_truth)

    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files", "img_name": "DMI_009_T2_56", "img_header": ground_header}
    given_lambda = 6e-3
    given_eta = 4e-3
    op = anic.AnatomicReconstructor(structural_data, (11,11), given_lambda, given_eta, 8000, True, save_options)
    # _op.low_res_data = _low_res_data
    # variables = [9e-3, 6e-3, 4e-3]
    # diff_images(_ground_truth, _op, variables, "lambda")
    recon_T2 = op(low_res_data)
    # print(find_mse(_ground_truth, _recon))


    FLAIR_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t2_flair.nii.gz"))
    FLAIR_data = FLAIR_header.get_fdata()[:, :, 56]
    FLAIR_data = normalize_matrix(FLAIR_data)
    op.anatomical_data =FLAIR_data
    op.img_name = "DMI_009_FLAIR_56"
    op.img_header = FLAIR_header
    recon_FLAIR = op(low_res_data)

    T1c_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t1.nii.gz"))
    T1c_data = T1c_header.get_fdata()[:, :, 56]
    T1c_data = normalize_matrix(T1c_data)
    op.anatomical_data =T1c_data
    op.img_name = "DMI_009_T1_56"
    op.img_header = T1c_header
    recon_T1c = op(low_res_data)

    if xp.__name__ == "cupy":
            ground_truth = ground_truth.get()
            low_res_data = low_res_data.get()
            recon_T2 = recon_T2.get()
            recon_FLAIR = recon_FLAIR.get()
            recon_T1c = recon_T1c.get()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))
    ax.ravel()[0].imshow(ground_truth, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[0].set_title("ground truth")
    ax.ravel()[0].axis("off")
    ax.ravel()[1].imshow(recon_FLAIR, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[1].set_title("FLAIR")
    ax.ravel()[1].axis("off")
    ax.ravel()[2].imshow(recon_T1c, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[2].set_title("T1")
    ax.ravel()[2].axis("off")
    ax.ravel()[3].imshow(recon_T2, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[3].set_title("T2")
    ax.ravel()[3].axis("off")
    fig.show()

def compare_downsamplings():
    """
    Compare the resolutions of different downsampling resolutions
    """
    ground_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/DMI_patient_9_ds_2_gm_3.0_wm_1.0_tumor_5.0_ed_2.0_noise_0.0/dmi_gt.nii.gz"))
    ground_truth = ground_header.get_fdata()[:, :, 56, 0]
    ground_truth = xp.array(normalize_matrix(ground_truth))
    structural_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t2_flair.nii.gz"))
    structural_data = structural_header.get_fdata()[:, :, 56]
    structural_data = normalize_matrix(structural_data)

    DS2_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/DMI_patient_9_ds_2_gm_3.0_wm_1.0_tumor_5.0_ed_2.0_noise_0.0/dmi.nii.gz"))
    DS2_data = DS2_header.get_fdata()[:,:, 56, 0][::2, ::2]
    DS2_data = normalize_matrix(DS2_data)

    save_options = {"given_path": "project_data/BraTS_DS_Experiments_Reconstructions/Nifity_Files", "img_name": "DMI9_56DS2_FLAIR", "img_header": ground_header}
    given_lambda = 6e-3
    given_eta = 2e-2
    op = anic.AnatomicReconstructor(structural_data, given_lambda, given_eta, 20000, True, save_options)
    op.given_lambda = 1e-2
    op.given_eta = 1e-2
    recon_DS2 = op(DS2_data)


    DS5_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/DMI_patient_9_ds_5_gm_3.0_wm_1.0_tumor_5.0_ed_2.0_noise_0.0/dmi.nii.gz"))
    DS5_data = DS5_header.get_fdata()[:, :, 56, 0][::5, ::5]
    DS5_data = normalize_matrix(DS5_data)
    op.img_name = "DMI9_56DS5_FLAIR"
    op.given_lambda = 6e-3
    op.given_eta = 9e-3
    recon_DS5 = op(DS5_data)

    # DS10_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/DMI_patient_9_ds_10_gm_3.0_wm_1.0_tumor_5.0_ed_2.0_noise_0.0/dmi.nii.gz"))
    # DS10_data = DS10_header.get_fdata()[:, :, 56, 0][::10, ::10]
    # DS10_data = normalize_matrix(DS10_data)
    # op.img_name = "DMI9_56DS10_T2"
    # op.given_lambda = 6e-3
    # op.given_eta = 1e-3
    # recon_DS10 = op(DS10_data)

    DS11_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/DMI_patient_9_ds_11_gm_3.0_wm_1.0_tumor_5.0_ed_2.0_noise_0.0/dmi.nii.gz"))
    DS11_data = DS11_header.get_fdata()[:, :, 56, 0][::11, ::11]
    DS11_data = normalize_matrix(DS11_data)
    op.img_name = "DMI9_56DS11_FLAIR"
    op.given_lambda = 6e-3
    op.given_eta = 1e-3
    recon_DS11 = op(DS11_data)

    if xp.__name__ == "cupy":
            ground_truth = ground_truth.get()
            recon_DS2 = recon_DS2.get()
            recon_DS5 = recon_DS5.get()
            recon_DS11 = recon_DS11.get()

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,15))
    ax.ravel()[0].imshow(ground_truth, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[0].set_title("ground_truth")
    ax.ravel()[0].axis("off")
    ax.ravel()[1].imshow(DS2_data, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[1].set_title("4mm")
    ax.ravel()[1].axis("off")
    ax.ravel()[2].imshow(DS5_data, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[2].set_title("10mm")
    ax.ravel()[2].axis("off")
    ax.ravel()[3].imshow(DS11_data, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[3].set_title("22mm")
    ax.ravel()[3].axis("off")
    ax.ravel()[4].imshow(structural_data, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[4].set_title("structal")
    ax.ravel()[4].axis("off")
    ax.ravel()[5].imshow(recon_DS2, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[5].set_title("")
    ax.ravel()[5].axis("off")
    ax.ravel()[6].imshow(recon_DS5, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[6].set_title("")
    ax.ravel()[6].axis("off")
    ax.ravel()[7].imshow(recon_DS11, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[7].set_title("")
    ax.ravel()[7].axis("off")
    fig.tight_layout()
    fig.show()

def compare_3D_downsamplings():
    """
    Compare the resolutions of different downsampling resolutions
    """
    ground_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/tumor_0.5/DMI_patient_9_ds_11_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0/dmi_gt.nii.gz"))
    ground_truth = ground_header.get_fdata()[:, :, :, 0]
    ground_truth = xp.array(normalize_matrix(ground_truth))
    structural_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t1.nii.gz"))
    structural_data = structural_header.get_fdata()
    structural_data = xp.array(normalize_matrix(structural_data))
    

    DS2_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/tumor_0.5/DMI_patient_9_ds_2_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0/dmi.nii.gz"))
    DS2_data = DS2_header.get_fdata()[:, :, :, 0][::2, ::2, ::2]
    DS2_data = normalize_matrix(DS2_data)

    save_options = {"given_path": "project_data/BraTS_DS_Experiments_Reconstructions/Nifity_Files", "img_name": "DMI9_0.5DS2_T1", "img_header": ground_header}
    given_lambda = 9e-3
    given_eta = 2e-2
    op = anic.AnatomicReconstructor(structural_data, given_lambda, given_eta, 20000, True, save_options)
    op.given_lambda = 1e-2
    op.given_eta = 7e-3
    recon_DS2 = op(DS2_data)


    DS5_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/tumor_0.5/DMI_patient_9_ds_5_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0/dmi.nii.gz"))
    DS5_data = DS5_header.get_fdata()[:, :, :, 0][::5, ::5, ::5]
    DS5_data = normalize_matrix(DS5_data)
    op.img_name = "DMI9_0.5DS5_T1"
    op.given_lambda = 6e-3
    op.given_eta = 4e-4
    recon_DS5 = op(DS5_data)

    DS11_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/tumor_0.5/DMI_patient_9_ds_11_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0/dmi.nii.gz"))
    DS11_data = DS11_header.get_fdata()[:, :, :, 0][::11, ::11, ::11]
    DS11_data = normalize_matrix(DS11_data)
    op.img_name = "DMI9_0.5DS11_T1"
    op.given_lambda = 6e-3
    op.given_eta = 1e-3

    recon_DS11 = op(DS11_data)

    # DS10_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/tumor_0.5/DMI_patient_9_ds_10_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0/dmi.nii.gz"))
    # DS10_data = DS10_header.get_fdata()[:, :, :, 0][::10, ::10, ::10]
    down10 = spl.AverageDownsampling(ground_truth.shape, (10,10,10))
    DS10_data = down10(ground_truth)
    DS10_data = normalize_matrix(DS10_data)
    op.img_name = "HalfDMI9_0.5DS10_T1"
    op.given_lambda = 6e-3
    op.given_eta = 1e-3
    down2 = spl.AverageDownsampling(structural_data.shape, (2,2,2))
    structural_data = down2(structural_data)
    op.anatomical_data = structural_data
    recon_DS10 = op(DS10_data)

    SLICE= 56

    if xp.__name__ == "cupy":
            ground_truth = ground_truth.get()[:,:, SLICE]
            structural_data = structural_data.get()[:, :, SLICE//2]
            recon_DS2 = recon_DS2.get()[:, :, SLICE]
            recon_DS5 = recon_DS5.get()[:, :, SLICE]
            recon_DS10 = recon_DS10.get()[:, :, SLICE//2]
            recon_DS11 = recon_DS11.get()[:, :, SLICE]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(20,15))
    ax.ravel()[0].imshow(ground_truth, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[0].set_title("gt")
    ax.ravel()[0].axis("off")
    ax.ravel()[1].imshow(DS2_data[:, :, SLICE//2], vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[1].set_title("4mm")
    ax.ravel()[1].axis("off")
    ax.ravel()[2].imshow(DS5_data[:, :, SLICE//5], vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[2].set_title("10mm")
    ax.ravel()[2].axis("off")
    ax.ravel()[3].imshow(DS10_data[:, :, SLICE//10].get(), vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[3].set_title("20mm")
    ax.ravel()[3].axis("off")
    ax.ravel()[4].imshow(DS11_data[:, :, SLICE//11], vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[4].set_title("22mm")
    ax.ravel()[4].axis("off")
    ax.ravel()[5].imshow(structural_data, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[5].set_title("struct")
    ax.ravel()[5].axis("off")
    ax.ravel()[6].imshow(recon_DS2, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[6].set_title("2mm")
    ax.ravel()[6].axis("off")
    ax.ravel()[7].imshow(recon_DS5, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[7].set_title("2mm")
    ax.ravel()[7].axis("off")
    ax.ravel()[8].imshow(recon_DS10, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[8].set_title("4mm")
    ax.ravel()[8].axis("off")
    ax.ravel()[9].imshow(recon_DS11, vmin = 0, vmax = ground_truth.max(), cmap = 'Greys_r')
    ax.ravel()[9].set_title("2mm")
    ax.ravel()[9].axis("off")
    fig.tight_layout()
    fig.show()

# if __name__ == "__main__":
    # compare_3D_downsamplings()
    # compare_aquisitions()
    # _ground_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/tumor_0.5/DMI_patient_9_ds_11_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0/dmi_gt.nii.gz"))
    # _ground_truth = _ground_header.get_fdata()[:, :, :, 0]
    # _ground_truth = xp.array(normalize_matrix(_ground_truth))
    # _structural_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t1.nii.gz"))
    # _structural_data = _structural_header.get_fdata()
    # _structural_data = xp.array(normalize_matrix(_structural_data))

    # _low_res_data_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DS_Experiments/DMI_patient_9_ds_11_gm_3.0_wm_1.0_tumor_5.0_ed_2.0_noise_0.0/dmi.nii.gz"))
    # _low_res_data = _low_res_data_header.get_fdata()[:, :, 56, 0][::11, ::11]
    # _low_res_data = normalize_matrix(_low_res_data)


    # _save_options = {"given_path": "project_data/BraTS_DS_Experiments_Reconstructions/Nifity_Files", "img_name": "DMI9_56DS11_T1", "img_header": _ground_header}
    # _given_lambda = 6e-3
    # _given_eta = 3e-3
    # _op = anic.AnatomicReconstructor(_structural_data, _given_lambda, _given_eta, 20000, True, _save_options)
    # _op.low_res_data = _low_res_data
    # variables = [5e-2, 6e-3, 1e-4]
    # diff_images(_ground_truth, _op, variables, "lambda")
    # _recon = _op(_low_res_data)
    # # # print(find_mse(_ground_truth, _recon))

    # if xp.__name__ == "cupy":
    #         _ground_truth = _ground_truth.get()#[:, :, 56]
    #         _low_res_data = _low_res_data#[:, :, 11]
    #         _structural_data = _structural_data#[:, :,56]
    #         _recon = _recon.get()#[:, :, 56]

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))
    # ax.ravel()[0].imshow(_ground_truth, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[0].set_title("ground truth")
    # ax.ravel()[0].axis("off")
    # ax.ravel()[1].imshow(_low_res_data, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[1].set_title("low res")
    # ax.ravel()[1].axis("off")
    # ax.ravel()[2].imshow(_structural_data, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[2].set_title("structure")
    # ax.ravel()[2].axis("off")
    # ax.ravel()[3].imshow(_recon, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[3].set_title(f"Reconstruction_{_op.given_lambda}_{_op.given_eta}")
    # ax.ravel()[3].axis("off")
    # fig.show()
    # fig.savefig(r"project_data/BraTS_Noise_Experiments_Reconstructions/Composites/DMI_brain_optimal.png")