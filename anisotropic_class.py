"""
Implements subclass for Anatomically guided reconstruction using linear least
squares regressor from sigpy and other subclasses
Setters allow for dynamic setting
"""
from __future__ import annotations
from typing import Union
import pathlib
import math

try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np
import sigpy as sp
import nibabel as nib
import pandas as pd

import downsampling_subclass as spl
import projection_operator_subclass as proj


class AnatomicReconstructor():
    """
    class to facilate anatomic reconstruction, apply on low res data
    """
    def __init__(self, anatomical_data: xp.ndarray, given_lambda: float, given_eta: float, iter_num: int, normalize: bool = False, save_options: dict = None) -> None:
        """
        Pass in needed information to set up reconstruction

        Input save options to save images to a folder
        save_options={"given_path":, "img_data":, "img_header":, "stats":}
        given_path is the path to save the image to
        image_data is a dictionary that writes image parameters to a csv
            img_data={"pt_type":, "pt_id":, "dmi_type":, "dmi_settings":, "contrast_type":, "prior_res":, "dmi_res":, "noise_level":,
                        "noise_seed":}
        img_header is the Nifity header file that will be saved with the image
        """
        if normalize:
            self._anatomical_data = xp.array(normalize_matrix(anatomical_data))
        else:
            self._anatomical_data = xp.array(anatomical_data)

        self._img_data = {"lambda": given_lambda, "eta": given_eta, "iter_num": iter_num, "normalize": normalize}

        self._low_res_data = None
        self._downsampling_factor = None

        #Sets up saving settings and csv
        if save_options is not None:
            try:
                self._path = pathlib.Path(save_options["given_path"])
                self._img_header = save_options["img_header"]
                self._img_data.update(save_options["img_data"])
                self._stats = save_options["stats"]
            except KeyError as mal:
                raise ValueError(
                    f"malformed save_options input {save_options}, image failed to save") from mal
            
            self.saving = True
            filepath = pathlib.Path.joinpath(self._path, "recon_data.csv")
            if not pathlib.Path.exists(filepath):
                self._recon_csv = pd.DataFrame(columns=["pt_type", "pt_id", "dmi_type", "dmi_settings", "contrast_type", "prior_res", "dmi_res", "noise_level", "noise_seed", "lambda", "eta", "iter_num", "normalize", "mse", "perc.accuracy"])
            else:
                self._recon_csv = pd.read_csv(filepath, index_col=0)
        else:
            self.saving = False

    @property
    def anatomical_data(self):
        """
        returns anatomical data built into operator
        """
        return self._anatomical_data
    
    @property
    def low_res_data(self):
        """
        returns inputed low_res_data
        """
        if self._low_res_data is not None:
            return self._low_res_data
        raise ValueError(
            f"Input low_res_data into operator before calling it. Use .low_res_data setter or run {__name__} first")

    @property
    def downsampling_factor(self):
        """
        return set downsampling_factor
        """
        if self._downsampling_factor is not None:
            return self._downsampling_factor
        raise ValueError(
            "low_res_data must be inputed into operator before the downsampling factor is calculated and called"
            )

    @property
    def given_lambda(self):
        """
        return set lambda
        """
        return self._img_data["lambda"]

    @property
    def given_eta(self):
        """
        returns current eta value
        """
        return self._img_data["eta"]

    @property
    def iter_num(self):
        """
        returns level of iterations set
        """
        return self._img_data["iter_num"]

    @property
    def normalize(self):
        """
        sets if the AnatomicReconstructor will normalize data by default
        """
        return self._img_data["normalize"]

    @property
    def given_path(self):
        """
        returns the path to the folder where reconstructions are to be saved
        """
        try:
            return self._path
        except NameError as err:
            raise ValueError(
                "saving_options were not set so no path is declared") from err

    @property
    def img_header(self):
        """
        returns the current img header
        """
        try:
            return self._img_header
        except NameError as err:
            raise ValueError(
                "saving_options were not set so no header is stored") from err
        
    @property
    def img_data(self):
        """
        Returns the dictionary of saved img data
        """
        try:
            return self._img_data
        except NameError as err:
            raise ValueError(
                "saving_options were not set so no img_data is saved") from err

    @anatomical_data.setter
    def anatomical_data(self, value: xp.ndarray):
        """
        Allows for anatomical data to be adjusted 
        """
        if self._img_data["normalize"]:
            self._anatomical_data = xp.array(normalize_matrix(value))
        else:
            self._anatomical_data = xp.array(value)

    @low_res_data.setter
    def low_res_data(self, value: xp.ndarray):
        """
        Allows for low_res_data to be adjusted
        """
        if self._img_data["normalize"]:
            self._low_res_data = xp.array(normalize_matrix(value))
        else:
            self._low_res_data = xp.array(value)

    @given_lambda.setter
    def given_lambda(self, value: float):
        self._img_data["lambda"] = value

    @given_eta.setter
    def given_eta(self, value: float):
        self._img_data["eta"] = value

    @iter_num.setter
    def iter_num(self, value: int):
        self._img_data["iter_num"] = value

    @given_path.setter
    def given_path(self, value: str):
        self._path = value

    @img_header.setter
    def img_header(self, value):
        self._img_header = value

    @img_data.setter
    def img_data(self, value: dict):
        self._img_data = value

    def __call__(self, iarray) -> xp.ndarray:
        """
        Calls the AnatomicReconstructor Operator on the inputed array

        If saving_options are set then AnatomicReconstructor can pull from
        already constructed images
        """
        #normalizes input matrix if set
        if self._img_data["normalize"]:
            self._low_res_data = xp.array(normalize_matrix(iarray))
        else:
            self._low_res_data = xp.array(iarray)
        #Checks if low_res_data shape can be upscaled to the same size as the anatomical_data
        if all((elem_1 % elem_2) == 0 for elem_1, elem_2 in zip(self._anatomical_data.shape, self._low_res_data.shape)):
            self._downsampling_factor = tuple(elem_1 // elem_2 for elem_1, elem_2 in zip(self._anatomical_data.shape, self._low_res_data.shape))
        else:
            self._anatomical_data = pad_array(self._anatomical_data, self._low_res_data.shape)
            self._downsampling_factor = tuple(elem_1 // elem_2 for elem_1, elem_2 in zip(self._anatomical_data.shape, self._low_res_data.shape))
        if self.saving:
            filename = self.search_image()
            if filename is not None:
                img_header = nib.as_closest_canonical(nib.load(filename))
                return xp.array(img_header.get_fdata())
            reconstruction = self.run_reconstructor()
            if self._stats:
                stats = self.calculate_stats(reconstruction)
                self.img_data["mse"] = stats[0]
                self.img_data["perc.accuracy"] = stats[1]
            self.save_image(reconstruction)
        else:
            reconstruction = self.run_reconstructor()
        return xp.array(reconstruction)

    def run_reconstructor(self):
        """
        Runs the reconstructor algorithm and masks the result to prevent the MSE being influenced by unnessary
        background pixels
        """
        downsampler = spl.AverageDownsampling(
            self._anatomical_data.shape, self._downsampling_factor)
        downsampled = self._low_res_data
        gradient_op = sp.linop.FiniteDifference(self._anatomical_data.shape)
        projection_op = proj.ProjectionOperator(
            gradient_op.oshape, self._anatomical_data, eta=self._img_data["eta"])
        compose_op = sp.linop.Compose([projection_op, gradient_op])
        gproxy = sp.prox.L1Reg(compose_op.oshape, self._img_data["lambda"])

        alg = sp.app.LinearLeastSquares(
            downsampler, downsampled, proxg=gproxy, G=compose_op, max_iter=self._img_data["iter_num"])
        result = alg.run()
        masked_result = result * (self._anatomical_data > 0.0001)
        if xp.__name__ == "cupy":
            masked_result = masked_result.get()
        return masked_result

    def save_image(self, img):
        """
        Saves reconstruction to a folder that can be read from later
        """
        recon_img = nib.Nifti1Image(
            img, self.img_header.affine, self.img_header.header)
        filename = f"{len(self._recon_csv)}.nii"
        self._recon_csv = pd.concat([self._recon_csv, pd.DataFrame(self.img_data, index=[0])], ignore_index=True)
        csv_path = pathlib.Path.joinpath(self._path, "recon_data.csv")
        self._recon_csv.to_csv(csv_path)
        save_path = self._path.joinpath(filename)
        nib.save(recon_img, save_path)

    def search_image(self) -> Union[str, None]:
        """
        Checks if an image with the current settings has already been generated. 
        If so it returns the path otherwise it returns None
        """
        mask = pd.Series(True, index=self._recon_csv.index)
        attributes = self._recon_csv[["pt_type", "pt_id", "dmi_type", "dmi_settings", "contrast_type", "prior_res",
                                      "dmi_res", "noise_level", "noise_seed", "lambda", "eta", "iter_num", "normalize"]]
        for column in attributes.columns:
            if column not in self._img_data.keys():
                mask &= (attributes[column].isna())
            else:
                mask &= (attributes[column] == self._img_data[column])
        masked_rows = attributes[mask]
        if len(masked_rows) == 0:
            return None
        if len(masked_rows) == 1:
            return pathlib.Path.joinpath(self.given_path, f"{masked_rows.index[0]}.nii")
        raise ValueError(
            f"recon_csv at {self.given_path} has duplicate reconstructions at {masked_rows.index}")
    
    def calculate_stats(self, recon) -> float:
        """
        Calculates the mse and % accuracy of the reconstruction in the segmented area.
        return (mse, %accuracy)
        """
        #Ground_Truth
        gt_header = nib.as_closest_canonical(nib.load(
        f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_{self.img_data['pt_id']}/{self.img_data['dmi_type']}_pt{self.img_data['pt_id']}_vs_2_ds_6_{self.img_data['dmi_settings']}_noise_{self.img_data['noise_level']}_seed_{self.img_data['noise_seed']}/dmi_gt.nii.gz"))
        ground_truth = gt_header.get_fdata()[:, :, :, 0]
        ground_truth = xp.array(normalize_matrix(ground_truth)).get()

        #Segmentation
        seg_header = nib.as_closest_canonical(nib.load(
            f"project_data/BraTS_Data/DMI_Simulations/DMI/patient_{self.img_data['pt_id']}/2mm_seg.nii.gz"
        ))
        seg = seg_header.get_fdata()
        seg = xp.array(normalize_matrix(seg))

        if self.img_data["prior_res"] % 2 == 0:
            #Recon
            upscaling_factor = self.img_data["prior_res"]//2
            upsampler = spl.AverageUpsampling(seg.shape, [upscaling_factor, upscaling_factor, upscaling_factor])
            recon = upsampler(recon)

        else:
            #Recon
            upscaling_factor = 240//recon.shape[0]
            to_240 = spl.AverageUpsampling([240, 240, 240], [upscaling_factor, upscaling_factor, upscaling_factor])
            recon = xp.asarray(to_240(recon))
            downsampler = spl.AverageDownsampling(recon.shape, [2, 2, 2])
            recon = downsampler(recon)
            if xp.__name__ == "cupy":
                recon = recon.get()


        mask = (seg == 1).get()
        masked_ground_truth = ground_truth[mask]
        masked_recon = recon[mask]

        squared_diff = (masked_ground_truth - masked_recon) ** 2

        ideal_value = masked_ground_truth.mean()
        average_value = masked_recon.mean()
        perc_accuracy = np.abs(ideal_value - average_value)/ideal_value * 100

        if perc_accuracy > 100:
            print("ideal_value", ideal_value, "average_value", average_value, self._img_data)

        return (np.mean(squared_diff), perc_accuracy)


def normalize_matrix(matrix):
    """
    normalizes inputed matrix so that all of it's values range from 0 - 1
    """
    m_max = matrix.max()
    if m_max != 1.0:
        return matrix/m_max
    return matrix

def pad_array(matrix: xp.ndarray, input_shape: tuple[int]):
    """
    pads the inputed matrix with zeros so that it can be divided evenly by
    the inputed factor
    """
    #Calculates how many rows/columns of zeros need to be added to properly pad the matrix
    elem_pad = tuple(elem_2 - (elem_1 % elem_2) for elem_1, elem_2 in zip(matrix.shape, input_shape))
    #Splits the padding evenly around the original matrix so that the image stays in the middle
    pad_parameters = tuple((elem//2, math.ceil(elem/2)) for elem in elem_pad)
    return np.pad(matrix, pad_parameters, "constant")
