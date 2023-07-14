"""
Testing Nibabel
"""
import nibabel as nib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    _img1_header = nib.as_closest_canonical(nib.load(r"project_data\BraTS_Data\Noise_Experiments\DMI_patient_9_ds_11_gm_4.0_wm_1.0_tumor_6.0_noise_0.001\dmi_gt.nii.gz"))
    _ground_truth = _img1_header.get_fdata()
    print(_ground_truth.shape)
    _ground_truth = _ground_truth[:, :, 70, 1]
    print(_ground_truth.shape)

    img = plt.imshow(_ground_truth, "Greys_r", vmin=0, vmax=_ground_truth.max())
    plt.show()