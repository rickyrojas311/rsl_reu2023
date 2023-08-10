"""
Testing environment for reconstructing DMI simulations
"""
from __future__ import annotations
import sys

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage

import anisotropic_class as anic
from anisotropic_operator import normalize_matrix

sys.path.append("/home/ricky")
from pydeutmr.pydeutmr.data import simulation_wrapper
from pydeutmr.pydeutmr.models import downsample

def display_DMI_res():
    """
    Creates matplotlib visualization for reconstructions for different inital
    DMI resolutions
    """
    SLICE = 60 - 1
    dmi_reses = [4, 12, 24]

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
    turn_axis_off(ax)
    # Ground Truth
    gt_header = nib.as_closest_canonical(nib.load(
        "project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_2_ds_2_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi_gt.nii.gz"))
    ground_truth = gt_header.get_fdata()[:, :, :, 0]
    ground_truth = xp.array(normalize_matrix(ground_truth))
    if xp.__name__ == "cupy":
        ground_truth = ground_truth.get()[:, :, SLICE]
    # Structural Data
    structural_header = nib.as_closest_canonical(nib.load(
        "project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/2mm_t1.nii.gz"))
    structural_data = structural_header.get_fdata()
    structural_data = xp.array(normalize_matrix(structural_data))
    
    ax[0][0].imshow(ground_truth, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
    ax[0][0].set_title("GT/Struct")
    ax[0][0].set_ylabel("Glx Orig")
    ax[1][0].imshow(structural_data.get()[:, :, SLICE], vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
    ax[1][0].set_ylabel("Glx Recon")
    for i in range(4):
        if i > 0:
            dmi_res = dmi_reses[i - 1]
            ds_factor = dmi_res//2
            # Low Res Data
            low_res_data_header = nib.as_closest_canonical(nib.load(
                f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_2_ds_{ds_factor}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
            low_res_data = low_res_data_header.get_fdata()[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
            low_res_data = xp.array(normalize_matrix(low_res_data))
            for j in range(3):
                if j == 0:
                    if xp.__name__ == "cupy":
                            show_data = low_res_data.get()
                    else:
                        show_data = low_res_data
                    ax[j][i].imshow(show_data[:, :, SLICE//ds_factor], vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                    ax[j][i].set_title(f"{dmi_res}mm")
                    
                elif j == 1:
                    # Reconstructions
                    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                    "img_name": f"DMI9_Glx_2mmPrior_{dmi_res}mmDMI_0Noise_1234Seed", "img_header": gt_header}
                    oper = anic.AnatomicReconstructor(
                        structural_data, 6e-3, 1e-3, 20000, True, save_options)
                    recon = oper(low_res_data)
                    if xp.__name__ == "cupy":
                        recon = recon.get()
                    recon = recon[:, :, SLICE]
                    ax[j][i].imshow(
                        recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                else:
                    #Interpolations
                    if xp.__name__ == "cupy":
                        low_res_data = low_res_data.get()
                    interpolated_data = ndimage.zoom(low_res_data, (ds_factor, ds_factor, ds_factor), order=3)
                    interpolated_data = interpolated_data[:, :, SLICE]
                    ax[j][i].imshow(
                        interpolated_data, vmin=0, vmax=interpolated_data.max(), cmap='Greys_r')
                    
    fig.tight_layout()
    fig.show()

def display_prior_res():
    """
    Explores the difference in reconstruction quality based on the anatomical prior
    """
    SLICE = 60 - 1
    prior_reses = [2, 6, 4, 3, 2]
    dmi_reses = [12, 24]
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(12, 12))
    turn_axis_off(ax)
    for i in range(5):
        prior_res = prior_reses[i]
        # Ground Truth
        gt_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_{prior_res}_ds_{12//prior_res}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi_gt.nii.gz"))
        ground_truth = gt_header.get_fdata()[:, :, :, 0]
        ground_truth = xp.array(normalize_matrix(ground_truth))
        if xp.__name__ == "cupy":
            ground_truth = ground_truth.get()[:, :, (SLICE * 2)//prior_res]

        # Structural Data
        structural_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{prior_res}mm_t1.nii.gz"))
        structural_data = structural_header.get_fdata()
        structural_data = xp.array(normalize_matrix(structural_data))
        structural_data = structural_data

        for j in range(3):
            if j == 0:
                if i == 1:
                    ax[0][i].set_ylabel("Ground Truth")
                if i != 0:
                    ax[0][i].imshow(ground_truth, vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[0][i].set_title(f"{prior_res}mm")
                    
            else:
                dmi_res = dmi_reses[j - 1]
                ds_factor = dmi_res//prior_res
                # Low Res Data
                low_res_data_header = nib.as_closest_canonical(nib.load(
                    f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_{prior_res}_ds_{ds_factor}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
                low_res_data = low_res_data_header.get_fdata(
                )[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
                low_res_data = xp.array(normalize_matrix(low_res_data))

                if i == 0:
                    if j == 1:
                        ax[j][i].set_title("Low Res")
                    if xp.__name__ == "cupy":
                        show_data = low_res_data.get()
                    else:
                        show_data = low_res_data
                    if j == 1:
                        show_data = show_data[:, :, int(SLICE/120 * 20)]
                    else:
                        show_data = show_data[:, :, int(SLICE/120 * 10)]
                    ax[j][i].imshow(show_data, vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[j][i].set_ylabel(f"{dmi_res}mm")
                    
                else:
                    # Reconstructions
                    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                    "img_name": f"DMI9_Glx_{prior_res}mmPrior_{dmi_res}mmDMI_0Noise_1234Seed", "img_header": gt_header}
                    oper = anic.AnatomicReconstructor(
                        structural_data, 6e-3, 1e-3, 20000, True, save_options)
                    recon = oper(low_res_data)
                    if xp.__name__ == "cupy":
                        recon = recon.get()
                    recon = recon[:, :, (SLICE * 2)//prior_res]
                    ax[j][i].imshow(
                        recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                    
    fig.tight_layout()
    fig.show()

def display_prior_res_cont():
    """
    Expands to lower res priors for 24mm
    """
    SLICE = 60 - 1
    prior_reses = [2, 12, 8, 6]
    dmi_reses = 24
    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 20))
    turn_axis_off(ax)
    for i in range(4):
        prior_res = prior_reses[i]
        # Ground Truth
        gt_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_{prior_res}_ds_{24//prior_res}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi_gt.nii.gz"))
        ground_truth = gt_header.get_fdata()[:, :, :, 0]
        ground_truth = xp.array(normalize_matrix(ground_truth))
        if xp.__name__ == "cupy":
            ground_truth = ground_truth.get()[:, :, (SLICE * 2)//prior_res]

        # Structural Data
        structural_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{prior_res}mm_t1.nii.gz"))
        structural_data = structural_header.get_fdata()
        structural_data = xp.array(normalize_matrix(structural_data))

        for j in range(4):
            if j == 0:
                if i != 0:
                    ax[0][i].imshow(ground_truth, vmin=0,
                                    vmax=ground_truth.max(), cmap='Greys_r')
                    ax[0][i].set_title(f"{prior_res}mm")       
            elif j != 3 or i != 3:
                if j == 1:
                    # Low Res Data
                    dmi_res = dmi_reses
                    ds_factor = dmi_res//prior_res
                    low_res_data_header = nib.as_closest_canonical(nib.load(
                        f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_{prior_res}_ds_{ds_factor}_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/dmi.nii.gz"))
                    low_res_data = low_res_data_header.get_fdata()[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
                    low_res_data = xp.array(normalize_matrix(low_res_data))

                    if i == 0:
                        if xp.__name__ == "cupy":
                            show_data = low_res_data.get()
                        else:
                            show_data = low_res_data
                        show_data = show_data[:, :, int(SLICE/120 * 10)]
                        ax[j][i].imshow(show_data, vmin=0,
                                        vmax=ground_truth.max(), cmap='Greys_r')
                        ax[j][i].set_ylabel("24mm")
                        ax[j][i].set_title("Low Res")
                        

                    else:
                        #Reconstructions
                        save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                            "img_name": f"DMI9_Glx_{prior_res}mmPrior_{dmi_res}mmDMI_0Noise_1234Seed", "img_header": gt_header}
                        oper = anic.AnatomicReconstructor(
                            structural_data, 6e-3, 1e-3, 20000, True, save_options)
                        recon = oper(low_res_data)
                        if xp.__name__ == "cupy":
                            recon = recon.get()
                        recon = recon[:, :, (SLICE * 2)//prior_res]
                        ax[j][i].imshow(
                            recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                        

                else:
                    if i != 0:
                        #Reconstructions off reconstructions, needs higher res structural data
                        new_res = prior_res//((j - 1) * 2)
                        save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                        "img_name": f"DMI9_Glx_{prior_res}mmPrior_{dmi_res}mmDMI_0Noise_1234Seed_{j - 1}", "img_header": gt_header}
                        structural_header = nib.as_closest_canonical(nib.load(
                                            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{new_res}mm_t1.nii.gz"))
                        structural_data = structural_header.get_fdata()
                        structural_data = xp.array(normalize_matrix(structural_data))
                        oper = anic.AnatomicReconstructor(
                            structural_data, 6e-3, 1e-3, 20000, True, save_options)
                        recon = oper(low_res_data)
                        if xp.__name__ == "cupy":
                            recon = recon.get()
                        recon = recon[:, :, (SLICE * 2)//new_res]
                        ax[j][i].imshow(
                            recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                        
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
        "project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Glx_pt9_vs_2_ds_6_gm_3.0_wm_1.0_tumor_0.5_ed_2.0_noise_0_seed_1234/",
        "project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/Lac_pt9_vs_2_ds_6_gm_0.3_wm_0.1_tumor_3.0_ed_1.0_noise_0_seed_1234/"
    ]
    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(20, 12))
    turn_axis_off(ax)
    for i in range(5):
        if i > 1:
            # Structural Data
            structural_header = nib.as_closest_canonical(nib.load(
                f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/2mm_{contrasts[i - 2]}.nii.gz"))
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
                    ax[0][2].set_ylabel("Contrasts")
                

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
                    ax[1][i].set_ylabel("Glx")
                    ax[1][i].set_title("Ground Truth")
                    ax[2][i].set_ylabel("Lac")
                    

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
                        ax[1][i].set_title("Low Res")
                        

                    else:
                        # Reconstructions
                        save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                        "img_name": f"DMI9_{dmi_type}_2mmPrior_12mmDMI_0Noise_1234Seed_{contrasts[i -2]}", "img_header": gt_header}
                        oper = anic.AnatomicReconstructor(
                            structural_data, 6e-3, 1e-3, 20000, True, save_options)
                        recon = oper(low_res_data)
                        if xp.__name__ == "cupy":
                            recon = recon.get()
                        recon = recon[:, :, SLICE]
                        ax[j][i].imshow(
                            recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                        
    fig.tight_layout()
    fig.show()

def display_noise_effect(noise_levels: list[int], dmi_type: str, prior_res: int, dmi_res: int, display_slice:int):
    """
    Examines how noise can affect the reconstruction
    """
    #Set Parameters
    if dmi_type == "Glx":
        dmi_settings = "gm_3.0_wm_1.0_tumor_0.5_ed_2.0"
    elif dmi_type == "Lac":
        dmi_settings = "gm_0.3_wm_0.1_tumor_3.0_ed_1.0"
    else:
        raise ValueError(f"dmi_type must be easier Glx or Lac not {dmi_type}")
    display_slice -= 1
    if dmi_res % prior_res != 0:
        raise ValueError(f"dmi_res {dmi_res} should even scale into prior_res {prior_res}")
    if len(noise_levels) > 4:
        raise ValueError(f"Can only display up to 4 noise levelsnot {len(noise_levels)}")
    ds_factor = dmi_res//prior_res
    
    #Create Plot
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 7))
    turn_axis_off(ax)
    # Ground Truth
    gt_header = nib.as_closest_canonical(nib.load(
        f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{dmi_type}_pt9_vs_{prior_res}_ds_{ds_factor}_{dmi_settings}_noise_0.2_seed_1234/dmi_gt.nii.gz"))
    ground_truth = gt_header.get_fdata()[:, :, :, 0]
    ground_truth = xp.array(normalize_matrix(ground_truth))
    if xp.__name__ == "cupy":
        ground_truth = ground_truth.get()[:, :, (display_slice * 2)//prior_res]
    # Structural Data
    structural_header = nib.as_closest_canonical(nib.load(
        f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{prior_res}mm_t1.nii.gz"))
    structural_data = structural_header.get_fdata()
    structural_data = xp.array(normalize_matrix(structural_data))
    
    ax[0][0].imshow(ground_truth, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
    ax[0][0].set_title("GT/Struct")
    ax[0][0].set_ylabel(f"{dmi_type} Orig")
    ax[1][0].imshow(structural_data.get()[:, :, (display_slice * 2)//prior_res], vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
    ax[1][0].set_ylabel(f"{dmi_type} Recon")
    for i in range(5):
        if i > 0:
            noise = noise_levels[i - 1]
            # Low Res Data
            low_res_data_header = nib.as_closest_canonical(nib.load(
                f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{dmi_type}_pt9_vs_{prior_res}_ds_{ds_factor}_{dmi_settings}_noise_{noise}_seed_1234/dmi.nii.gz"))
            low_res_data = low_res_data_header.get_fdata()[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
            low_res_data = xp.array(normalize_matrix(low_res_data))
            for j in range(2):
                if j == 0:
                    if xp.__name__ == "cupy":
                            show_data = low_res_data.get()
                    else:
                        show_data = low_res_data
                    ax[j][i].imshow(show_data[:, :, (display_slice * 2)//prior_res//ds_factor], vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                    ax[j][i].set_title(f"{noise}")
                    
                else:
                    # Reconstructions
                    save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                    "img_name": f"DMI9_{dmi_type}_{prior_res}mmPrior_{dmi_res}mmDMI_{noise}Noise_1234Seed", "img_header": gt_header}
                    oper = anic.AnatomicReconstructor(
                        structural_data, 6e-3, 1e-3, 20000, True, save_options)
                    recon = oper(low_res_data)
                    if xp.__name__ == "cupy":
                        recon = recon.get()
                    recon = recon[:, :, (display_slice * 2)//prior_res]
                    ax[j][i].imshow(
                        recon, vmin=0, vmax=ground_truth.max(), cmap='Greys_r')
                    
    fig.tight_layout()
    fig.show()

def display_noise_stats(noise_level: int, dmi_type: str, display_slice: int, num_iter: int = 20,):
    """
    Examines how noise can affect the reconstruction by generating 20 different noise patterns and displaying a standard deviation
    """
    #Set Parameters
    if dmi_type == "Glx":
        dmi_settings = "gm_3.0_wm_1.0_tumor_0.5_ed_2.0"
    elif dmi_type == "Lac":
        dmi_settings = "gm_0.3_wm_0.1_tumor_3.0_ed_1.0"
    else:
        raise ValueError(f"dmi_type must be easier Glx or Lac not {dmi_type}")
    display_slice -= 1
    reses = [(24, 6), (12, 2)]
    # reses = [(24, 6)]

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15,15))
    turn_axis_off(ax)
    for j, res in enumerate(reses):
        dmi_res = res[0]
        prior_res = res[1]
        ds_factor = dmi_res//prior_res
        #Ground Truth
        gt_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{dmi_type}_pt9_vs_{prior_res}_ds_{ds_factor}_{dmi_settings}_noise_0.2_seed_1234/dmi_gt.nii.gz"))
        ground_truth = gt_header.get_fdata()[:, :, :, 0]
        ground_truth = xp.array(normalize_matrix(ground_truth))

        #Structural Data
        structural_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{prior_res}mm_t1.nii.gz"))
        structural_data = structural_header.get_fdata()
        structural_data = xp.array(normalize_matrix(structural_data))

        settings = {"i_type": dmi_type, "dmi_res": dmi_res, "prior_res": prior_res,
                    "brats_path": "/home/ricky/rsl_reu2023/project_data/BraTS_Data/DMI_Simulations",
                    "brats_num": 9, "noise": noise_level}
        np.random.seed(100)
        seeds = np.random.randint(1000, 9000, num_iter)

        #Noiseless recon
        noiseless_data_header = nib.as_closest_canonical(nib.load(
                    f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{dmi_type}_pt9_vs_{prior_res}_ds_{ds_factor}_{dmi_settings}_noise_0_seed_1234/dmi.nii.gz"))
        noiseless_data = noiseless_data_header.get_fdata()[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
        noiseless_data = xp.array(normalize_matrix(noiseless_data))
        noiseless_data = xp.array(normalize_matrix(noiseless_data))
        save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                    "img_name": f"DMI9_{dmi_type}_{prior_res}mmPrior_{dmi_res}mmDMI_0Noise_1234Seed", "img_header": gt_header}
        oper = anic.AnatomicReconstructor(
                structural_data, 6e-3, 1e-3, 20000, True, save_options)
        noiseless_recon = oper(noiseless_data)

        recons = xp.zeros((len(seeds),) + structural_data.shape)
        for i, seed in enumerate(seeds):
            settings["seed"] = seed
            simulation_wrapper(settings)

            #Noise Low Res Data
            noiseless_data_header = nib.as_closest_canonical(nib.load(
                    f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_9/{dmi_type}_pt9_vs_{prior_res}_ds_{ds_factor}_{dmi_settings}_noise_{noise_level}_seed_{seed}/dmi.nii.gz"))
            noiseless_data = noiseless_data_header.get_fdata()[:, :, :, 0][::ds_factor, ::ds_factor, ::ds_factor]
            noiseless_data = xp.array(normalize_matrix(noiseless_data))

            save_options = {"given_path": "project_data/BraTS_Reconstructions/Nifity_Files",
                                        "img_name": f"DMI9_{dmi_type}_{prior_res}mmPrior_{dmi_res}mmDMI_{noise_level}Noise_{seed}Seed", "img_header": gt_header}
            oper = anic.AnatomicReconstructor(
                    structural_data, 6e-3, 1e-3, 20000, True, save_options)
            noiseless_recon = oper(noiseless_data)
            recons[i,...] = noiseless_recon

        mean_recon = np.mean(recons, 0)

        stdev_recon = xp.zeros(mean_recon.shape)
        for i in range(recons.shape[0]):
            stdev_recon += (recons[i,...] - mean_recon) ** 2
        stdev_recon = (stdev_recon/recons.shape[0]) ** 0.5

        mean_recon = mean_recon[:, :, (display_slice * 2)//prior_res]
        stdev_recon = stdev_recon[:, :, (display_slice * 2)//prior_res]
        noiseless_recon = noiseless_recon[:, :, (display_slice * 2)//prior_res]
        if xp.__name__ == "cupy":
            mean_recon = mean_recon.get()
            stdev_recon = stdev_recon.get()
            noiseless_recon = noiseless_recon.get()
            ground_truth = ground_truth.get()[:, :, (display_slice * 2)//prior_res]
        ax[j][0].imshow(ground_truth, "Greys_r",  vmin=0, vmax=ground_truth.max())
        ax[j][1].imshow(noiseless_recon, "Greys_r",  vmin=0, vmax=ground_truth.max())
        ax[j][2].imshow(mean_recon, "Greys_r", vmin=0, vmax=ground_truth.max())
        ax[j][3].imshow(stdev_recon, "Greys_r", vmin=0, vmax=ground_truth.max())
        ax[j][4].imshow(mean_recon - noiseless_recon, "seismic", vmin=-ground_truth.max(), vmax=ground_truth.max())
        ax[0][0].set_title("Ground Truth")
        ax[0][0].set_ylabel("24mm")
        ax[0][1].set_title("No Noise")
        ax[1][0].set_ylabel("12mm")
        ax[0][2].set_title("Mean Noise")
        ax[0][3].set_title("Std Dev")
        ax[0][4].set_title("Error")

    fig.tight_layout()
    fig.show()

def display_in_vivo():
    """
    Displays the in-vivo data collected from patient C
    """
    display_slice = 68 - 1
    prior_res = 3
    dmi_res = 15
    ds_factor = 15//3
    contrasts = ["Glx", "Lac", "Glx/Lac Ratio"]

    # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 7))
    fig, ax = plt.subplots()
    # turn_axis_off(ax)
    # Structural Data
    structural_header = nib.as_closest_canonical(nib.load(
        "project_data/In_Vivo/Patient C/t1.nii.gz"))
    structural_data = structural_header.get_fdata()
    structural_data = xp.array(normalize_matrix(structural_data))
    print(structural_data.shape)
    structural_data = downsample(downsample(downsample(structural_data, prior_res, axis=0),
                    prior_res, axis=1), prior_res, axis=2)
    print("struct", structural_data.shape)

    #Low Res Data
    low_res_data_header = nib.as_closest_canonical(nib.load(
        # "project_data/In_Vivo/Patient C/lac_glx_ratio.nii.gz"))
        "project_data/In_Vivo/Patient C/glx.nii.gz"))
        # "project_data/In_Vivo/Patient C/lac.nii.gz"))
    low_res_data = low_res_data_header.get_fdata()
    low_res_data = xp.array(normalize_matrix(low_res_data))
    low_res_data = downsample(downsample(downsample(low_res_data, prior_res, axis=0),
                    prior_res, axis=1), prior_res, axis=2)
    low_res_data = low_res_data[::ds_factor, ::ds_factor, ::ds_factor]
    print("low_res", low_res_data.shape)

    #Reconstruction
    save_options = {"given_path": "project_data/In_Vivo_Reconstructions",
                                    "img_name": "PtC_Glx_vs_3_ds_5", "img_header": structural_header}
    oper = anic.AnatomicReconstructor(
        structural_data, 6e-3, 1e-3, 20000, True, save_options)
    recon = oper(low_res_data)
    if xp.__name__ == "cupy":
        structural_data = structural_data.get()
        low_res_data = low_res_data.get()
        recon = recon.get()
    structural_data = structural_data[(display_slice * 2)//prior_res, :, :]
    low_res_data = low_res_data[(display_slice * 2)//prior_res//ds_factor, :, :]
    recon = recon[(display_slice * 2)//prior_res, :, :]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 20))
    turn_axis_off(ax)
    ax[0].imshow(structural_data, "Greys_r", vmin=0, vmax=structural_data.max())
    ax[0].set_title("structural")
    ax[1].imshow(low_res_data, "Greys_r", vmin=0, vmax=structural_data.max())
    ax[1].set_title("Glx")
    ax[2].imshow(recon, vmin=0, vmax=structural_data.max(), cmap='Greys_r')
    ax[2].set_title("Recon")
                    
    fig.tight_layout()
    fig.show()

def turn_axis_off(axes: np.ndarray):
    for ax in axes.flat:
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)


if __name__ == "__main__":
    display_DMI_res()
    # display_prior_res()
    # display_prior_res_cont()
    # display_MR_contrast()
    # Lac = [0, 0.233, 0.33, 0.466]
    # Glx = [0, 0.1414, 0.2, 0.2828]
    # display_noise_effect(Lac, "Lac", 6, 24, 60)
    # display_noise_effect(Glx, "Glx", 6, 24, 60)
    # display_noise_stats(0.2, "Glx", 60)
    # display_noise_stats(0.33, "Lac", 60)
    # display_in_vivo()
