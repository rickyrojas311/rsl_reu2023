"""
Testing environment for reconstructing DMI simulations
"""
from __future__ import annotations
import math
import pathlib

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
import anisotropic_operator as anio

if __name__ is "__main__":
    _img1_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/Noise_Experiments/DMI_patient_9_ds_11_gm_4.0_wm_1.0_tumor_6.0_noise_0.001/dmi_gt.nii.gz"))
    _ground_truth = _img1_header.get_fdata()[:, :, 56, 0]
    _ground_truth = normalize_matrix(_ground_truth)
    _img2_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/Noise_Experiments/DMI_patient_9_ds_11_gm_4.0_wm_1.0_tumor_6.0_noise_0.001/dmi.nii.gz"))
    _low_res_data = _img2_header.get_fdata()[:, :, 56, 0]
    _low_res_data = _low_res_data[::11, ::11]
    _low_res_data = normalize_matrix(_low_res_data)
    _img3_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t2_flair.nii.gz"))
    _structural_data = _img3_header.get_fdata()[:, :, 56]
    _structural_data = xp.array(normalize_matrix(_structural_data))
    # _down = spl.AverageDownsampling(_structural_data.shape, (2, 2))
    # _structural_data = _down(_structural_data)

    # print(_low_res_data.shape, _structural_data.shape)

    save_options = {"given_path": r"project_data/BraTS_Noise_Expirements_Reconstructions", "img_header": _img1_header}
    _op = anic.AnatomicReconstructor(_structural_data, (11, 11), 0.007, 0.0015, 8000, True, save_options)
    # _op.low_res_data = _low_res_data
    # _op.given_lambda = sweep_lambda(_ground_truth, _op)
    # _op.given_eta = sweep_eta(_ground_truth, _op)
    # _recon = _op(_low_res_data)

    # img = plt.imshow(_recon, "Greys_r", vmin=0, vmax=_ground_truth.max())
    # filename = f"{_low_res_data.ndim}D_lambda-{_op.given_lambda}_eta-{_op.given_eta}_iter-{_op.max_iter}"
    # if _op.normalize:
    #     filename += "_norm"
    # filename += ".png"
    # search_path = pathlib.Path("project_data/BraTS_Noise_Expirements_Reconstructions/Final Build").joinpath(filename)
    # plt.savefig(search_path)
    # plt.show()

    # _recon = _op(_low_res_data)
    # img = plt.imshow(_recon, "Greys_r", vmin=0, vmax=_ground_truth.max())
    # plt.show()

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20,15))
    # ax.ravel()[0].imshow(_ground_truth, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[0].set_title("ground truth")
    # ax.ravel()[0].axis("off")
    # ax.ravel()[1].imshow(_low_res_data, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[1].set_title("low res")
    # ax.ravel()[1].axis("off")
    # ax.ravel()[2].imshow(_structural_data.get(), vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[2].set_title("structure")
    # ax.ravel()[2].axis("off")
    # ax.ravel()[3].imshow(_recon, vmin = 0, vmax = _ground_truth.max(), cmap = 'Greys_r')
    # ax.ravel()[3].set_title("Reconstruction")
    # ax.ravel()[3].axis("off")
    # fig.show()
    # fig.savefig(r"project_data\BraTS_Noise_Expirements_Reconstructions\patient_9_all_data_side_by+reconstruction")