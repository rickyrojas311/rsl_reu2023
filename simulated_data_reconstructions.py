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
    # Ground Truths
    glx_header = nib.as_closest_canonical(nib.load(
        r"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_ds_2_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi_gt.nii.gz"))
    glx_gt = glx_header.get_fdata()[:, :, :, 0]
    glx_gt = xp.array(normalize_matrix(glx_gt))
    # lac_header = nib.as_closest_canonical(nib.load(r"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_ds_2_gm_0.3_wm_0.1_tumor_3.0_ed_1.0_noise_0_seed_1234/dmi_gt.nii.gz"))
    # lac_gt = lac_header.get_fdata()[:, :, :, 0]
    # lac_gt = xp.array(normalize_matrix(lac_gt))

    # Structural Data
    structural_header = nib.as_closest_canonical(
        nib.load(r"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/t1.nii.gz"))
    structural_data = structural_header.get_fdata()
    structural_data = xp.array(normalize_matrix(structural_data))

    # Low Res Data
    mm4_header = nib.as_closest_canonical(nib.load(
        r"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_ds_2_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
    data_4mm = mm4_header.get_fdata()[:, :, :, 0][::2, ::2, ::2]
    data_4mm = xp.array(normalize_matrix(data_4mm))
    mm12_header = nib.as_closest_canonical(nib.load(
        r"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_ds_6_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
    data_12mm = mm12_header.get_fdata()[:, :, :, 0][::6, ::6, ::6]
    data_12mm = xp.array(normalize_matrix(data_12mm))
    mm24_header = nib.as_closest_canonical(nib.load(
        r"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_ds_12_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
    data_24mm = mm24_header.get_fdata()[:, :, :, 0][::12, ::12, ::12]
    data_24mm = xp.array(normalize_matrix(data_24mm))

    # Operator set up
    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                    "img_name": "", "img_header": glx_header}
    oper = anic.AnatomicReconstructor(
        structural_data, 1e-2, 7e-3, 20000, True, save_options)

    # Reconstructions
    oper.img_name = "DMI9_Glx_4mm"
    recon_4mm = oper(data_4mm)

    oper.img_name = "DMI9_Glx_12mm"
    oper.given_lambda = 6e-3
    oper.given_eta = 4e-4
    recon_12mm = oper(data_12mm)

    oper.img_name = "DMI9_Glx_24mm"
    oper.given_lambda = 6e-3
    oper.given_eta = 1e-3
    recon_24mm = oper(data_24mm)

    SLICE = 60 - 1

    if xp.__name__ == "cupy":
        glx_gt = glx_gt.get()[:, :, SLICE]
        lac_gt = lac_gt.get()[:, :, SLICE]
        structural_data = structural_data.get()[:, :, SLICE]
        recon_4mm = recon_4mm.get()[:, :, SLICE]
        recon_12mm = recon_12mm.get()[:, :, SLICE]
        recon_24mm = recon_24mm.get()[:, :, SLICE]

    # Create Image
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(20, 15))
    ax = ax.ravel()
    ax[0].imshow(glx_gt, vmin=0, vmax=glx_gt.max(), cmap='Greys_r')
    ax[0].set_ylabel("Glx Orig")
    ax[0].set_title("GT/Struct")
    ax[0].axis("off")
    ax[1].imshow(data_4mm[:, :, SLICE//2].get(), vmin=0,
                 vmax=glx_gt.max(), cmap='Greys_r')
    ax[1].set_title("4mm")
    ax[1].axis("off")
    ax[2].imshow(data_12mm[:, :, SLICE//6].get(), vmin=0,
                 vmax=glx_gt.max(), cmap='Greys_r')
    ax[2].set_title("12mm")
    ax[2].axis("off")
    ax[3].imshow(data_24mm[:, :, SLICE//12].get(), vmin=0,
                 vmax=glx_gt.max(), cmap='Greys_r')
    ax[3].set_title("24mm")
    ax[3].axis("off")
    ax[4].imshow(structural_data, vmin=0, vmax=glx_gt.max(), cmap='Greys_r')
    ax[4].set_ylabel("Glx Recon")
    ax[4].axis("off")
    ax[5].imshow(recon_4mm, vmin=0, vmax=glx_gt.max(), cmap='Greys_r')
    ax[5].axis("off")
    ax[6].imshow(recon_12mm, vmin=0, vmax=glx_gt.max(), cmap='Greys_r')
    ax[6].axis("off")
    ax[7].imshow(recon_24mm, vmin=0, vmax=glx_gt.max(), cmap='Greys_r')
    ax[7].axis("off")
    fig.tight_layout()
    fig.show()


def display_prior_res():
    """
    Explores the difference in reconstruction quality based on the anatomical prior
    """
    SLICE = 60 - 1
    prior_reses = [2, 3, 4, 6]
    dmi_reses = [12, 24]
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 15))
    for i in range(5):
        if i != 4:
            prior_res = prior_reses[i]
            # Ground Truth
            gt_header = nib.as_closest_canonical(nib.load(
                f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_vs_{prior_res}_ds_{12//prior_res}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi_gt.nii.gz"))
            ground_truth = gt_header.get_fdata()[:, :, :, 0]
            ground_truth = xp.array(normalize_matrix(ground_truth))
            if xp.__name__ == "cupy":
                ground_truth = ground_truth.get()[:, :, (SLICE * 2)//prior_res]

            # Structural Data
            structural_header = nib.as_closest_canonical(nib.load(
                f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{prior_res}_t1.nii.gz"))
            structural_data = structural_header.get_fdata()
            structural_data = xp.array(normalize_matrix(structural_data))
            structural_data = structural_data

        for j in range(3):
            if j == 0:
                if i == 4:
                    ax[0][4].set_title(f"Low Res")
                    ax[0][4].axis("off")
                else:
                    ax[0][i].imshow(ground_truth, vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[0][i].set_title(f"{prior_res}mm")
                    ax[0][i].axis("off")
            else:
                dmi_res = dmi_reses[j - 1]
                ds_factor = dmi_res//prior_res
                # Low Res Data
                low_res_data_header = nib.as_closest_canonical(nib.load(
                    f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_vs_{prior_res}_ds_{ds_factor}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
                low_res_data = low_res_data_header.get_fdata(
                )[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
                low_res_data = xp.array(normalize_matrix(low_res_data))

                if i == 4:
                    if xp.__name__ == "cupy":
                        show_data = low_res_data.get()
                    else:
                        show_data = low_res_data
                    if j == 1:
                        show_data = show_data[:, :, int(SLICE/120 * 20)]
                    else:
                        show_data = show_data[:, :, int(SLICE/120 * 10)]
                    ax[j][4].imshow(show_data, vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[j][4].axis("off")
                else:
                    # Reconstructions
                    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                    "img_name": f"DMI9_Glx_{prior_res}mmPrior_{dmi_res}mmDMI", "img_header": gt_header}
                    oper = anic.AnatomicReconstructor(
                        structural_data, 6e-3, 1e-3, 20000, True, save_options)
                    recon = oper(low_res_data)
                    if xp.__name__ == "cupy":
                        recon = recon.get()
                    recon = recon[:, :, (SLICE * 2)//prior_res]
                    ax[j][i].imshow(
                        recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                    ax[j][i].axis("off")
    fig.tight_layout()
    fig.show()


def display_MR_contrast():
    """
    Displays the influence the underlying structural contrast has on the final image
    """
    SLICE = 60 - 1
    contrasts = ["t1", "t2", "t2_flair"]
    dmi_types = ["Glx", "Lac"]
    dmi_paths = [
        "project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_vs_2_ds_6_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/",
        "project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/DMI_patient_9_ds_6_gm_0.3_wm_0.1_tumor_3.0_ed_1.0_noise_0_seed_1234/"
    ]
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))
    for i in range(5):
        if i > 1:
            # Structural Data
            structural_header = nib.as_closest_canonical(nib.load(
                f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/2_{contrasts[i - 2]}.nii.gz"))
            structural_data = structural_header.get_fdata()
            structural_data = xp.array(normalize_matrix(structural_data))
            structural_data = structural_data

        for j in range(3):
            # Top Row shows structural data
            if j == 0:
                if i > 1:
                    ax[0][i].imshow(structural_data.get()[:, :, SLICE], vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[0][i].set_title(f"{contrasts[i - 2]}")
                ax[0][i].axis("off")

            # Second Row shows Glx, Third shows Lac
            else:
                file_path = dmi_paths[j - 1]
                dmi_type = dmi_types[j - 1]

                # Ground Truth
                gt_header = nib.as_closest_canonical(
                    nib.load(f"{file_path}dmi_gt.nii.gz"))
                ground_truth = gt_header.get_fdata()[:, :, :, 0]
                ground_truth = xp.array(normalize_matrix(ground_truth))
                if xp.__name__ == "cupy":
                    ground_truth = ground_truth.get()[:, :, SLICE]

                if i == 0:
                    ax[j][i].imshow(ground_truth, vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[j][i].axis("off")

                else:
                    # Low Res Data
                    low_res_data_header = nib.as_closest_canonical(
                        nib.load(f"{file_path}dmi.nii.gz"))
                    low_res_data = low_res_data_header.get_fdata()[
                        :, :, :, 0][::6, ::6, ::6]
                    low_res_data = xp.array(normalize_matrix(low_res_data))

                    if i == 1:
                        if xp.__name__ == "cupy":
                            show_data = low_res_data.get()
                        show_data = show_data[:, :, SLICE//6]
                        ax[j][i].imshow(show_data, vmin=0,
                                        vmax=ground_truth.max(), cmap='Greys_r')
                        ax[j][i].axis("off")

                    else:
                        # Reconstructions
                        save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                        "img_name": f"DMI9_{dmi_type}_2mmPrior_12mmDMI_{contrasts[i -2]}", "img_header": gt_header}
                        oper = anic.AnatomicReconstructor(
                            structural_data, 6e-3, 1e-3, 20000, True, save_options)
                        recon = oper(low_res_data)
                        if xp.__name__ == "cupy":
                            recon = recon.get()
                        recon = recon[:, :, SLICE]
                        ax[j][i].imshow(
                            recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                        ax[j][i].axis("off")
    fig.tight_layout()
    fig.show()


if __name__ == "__main__":
    display_MR_contrast()
