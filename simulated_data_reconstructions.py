"""
Testing environment for reconstructing DMI simulations
"""
from __future__ import annotations

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import anisotropic_class as anic
from anisotropic_operator import normalize_matrix

def display_DMI_res():
    """
    Creates matplotlib visualization for reconstructions for different inital
    DMI resolutions
    """
    #Ground Truths
    glx_header = nib.as_closest_canonical(nib.load(r""))
    glx_gt = glx_header.get_fdata()[:, :, :, 0]
    glx_gt = xp.array(normalize_matrix(glx_gt))
    lac_header = nib.as_closest_canonical(nib.load(r""))
    lac_gt = lac_header.get_fdata()[:, :, :, 0]
    lac_gt = xp.array(normalize_matrix(lac_gt))

    #Structural Data
    structural_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/MRI_2mm/t1.nii.gz"))
    structural_data = structural_header.get_fdata()
    structural_data = xp.array(normalize_matrix(structural_data))

    #Low Res Data
    mm4_header = nib.as_closest_canonical(nib.load(r""))
    data_4mm = mm4_header.get_fdata()[:, :, :, 0][::2, ::2, ::2]
    data_4mm = xp.array(normalize_matrix(data_4mm))
    mm12_header = nib.as_closest_canonical(nib.load(r""))
    data_12mm = mm12_header.get_fdata()[:, :, :, 0][::6, ::6, ::6]
    data_12mm = xp.array(normalize_matrix(data_12mm))
    mm24_header = nib.as_closest_canonical(nib.load(r""))
    data_24mm = mm24_header.get_fdata()[:, :, :, 0][::12, ::12, ::12]
    data_24mm = xp.array(normalize_matrix(data_24mm))

    #Operator set up
    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files", "img_name": "", "img_header": glx_header}
    oper = anic.AnatomicReconstructor(structural_data, 1e-2, 7e-3, 20000, True, save_options)

    #Reconstructions
    oper.img_name = ""
    recon_4mm = oper(data_4mm)

    oper.img_name = ""
    oper.given_lambda = 6e-3
    oper.given_eta = 4e-4
    recon_12mm = oper(data_12mm)

    oper.img_name = ""
    oper.given_lambda = 6e-3
    oper.given_eta = 1e-3
    recon_24mm = oper(data_24mm)

    SLICE= 56

    if xp.__name__ == "cupy":
            glx_gt = glx_gt.get()[:,:, SLICE]
            lac_gt = lac_gt.get()[:,:, SLICE]
            structural_data = structural_data.get()[:, :, SLICE]
            recon_4mm = recon_4mm.get()[:, :, SLICE]
            recon_12mm = recon_12mm.get()[:, :, SLICE]
            recon_24mm = recon_24mm.get()[:, :, SLICE]

    #Create Image
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20,15))
    ax = ax.ravel()
    ax[0].imshow(structural_data, vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[0].set_ylabel("Glx Orig")
    ax[0].set_title("GT")
    ax[0].axis("off")
    ax[1].imshow(data_4mm[:, :, SLICE//2], vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[1].set_title("4mm")
    ax[1].axis("off")
    ax[2].imshow(data_12mm[:, :, SLICE//6], vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[2].set_title("12mm")
    ax[2].axis("off")
    ax[3].imshow(data_24mm[:, :, SLICE//12], vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[3].set_title("24mm")
    ax[3].axis("off")
    ax[4].imshow(glx_gt, vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[4].set_ylabel("Glx Recon")
    ax[4].axis("off")
    ax[5].imshow(recon_4mm, vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[5].axis("off")
    ax[6].imshow(recon_12mm, vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[6].axis("off")
    ax[7].imshow(recon_24mm, vmin = 0, vmax = glx_gt.max(), cmap = 'Greys_r')
    ax[7].axis("off")
    fig.tight_layout()
    fig.show()

if __name__ == "__main__":
     display_DMI_res()