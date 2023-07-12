"""
Implements subclass for Anatomically guided reconstruction using linear least
squares regressor from sigpy and other subclasses
Setters allow for dynamic setting
"""
from typing import Union
import pathlib
import numpy as np
import sigpy as sp
import nibabel as nib

import downsampling_subclass as spl
import projection_operator_subclass as proj

class AnatomicReconstructor():
    """
    class to facilate anatomic reconstruction, apply on low res data
    """
    def __init__(self, anatomical_data: np.ndarray, downsampling_factor: tuple[int], given_lambda: float, given_eta: float, max_iter: int, normalize: bool=False, save_options: dict = None) -> None:
        """
        Pass in needed information to set up reconstruction

        Input save options to save images to a folder
        save_options={"given_path":, "img_header":}
        """
        if normalize:
            self._anatomical_data = normalize_matrix(anatomical_data)
        else:
            self._anatomical_data = anatomical_data

        self._downsampling_factor = downsampling_factor
        self._given_lambda = given_lambda
        self._given_eta = given_eta
        self._max_iter = max_iter
        self._normalize = normalize

        self._ground_truth = None

        if save_options is not None:
            try:
                self._path = pathlib.Path(save_options["given_path"])
                self._img_header = save_options["img_header"]
            except KeyError as mal:
                raise ValueError(f"malformed save_options input {save_options}, image failed to save") from mal
            self.saving = True
        else:
            self.saving = False


    @property
    def anatomical_data(self):
        """
        returns anatomical data built into operator
        """
        return self._anatomical_data

    @property
    def downsampling_factor(self):
        """
        return set downsampling_factor
        """
        return self._downsampling_factor

    @property
    def given_lambda(self):
        """
        return set lambda
        """
        return self._given_lambda

    @property
    def given_eta(self):
        """
        returns current eta value
        """
        return self._given_eta

    @property
    def max_iter(self):
        """
        returns level of iterations set
        """
        return self._max_iter

    @property
    def normalize(self):
        """
        sets if the AnatomicReconstructor will normalize data by default
        """
        return self._normalize

    @property
    def given_path(self):
        """
        returns the path to the folder where reconstructions are to be saved
        """
        try:
            return self._path
        except NameError as err:
            raise ValueError("saving_options were not set so no path is declared") from err
    
    @property
    def img_header(self):
        """
        returns the current img header
        """
        try:
            return self._img_header
        except NameError as err:
            raise ValueError("saving_options were not set so no header is stored") from err


    @anatomical_data.setter
    def anatomical_data(self, value: np.ndarray):
        """
        Allows for anatomical data to be adjusted 
        """
        if self._normalize:
            self._anatomical_data = normalize_matrix(value)
        else:
            self._anatomical_data = value

    @downsampling_factor.setter
    def downsampling_factor(self, value: tuple[int]):
        self._downsampling_factor = value

    @given_lambda.setter
    def given_lambda(self, value: float):
        self._given_lambda = value

    @given_eta.setter
    def given_eta(self, value: float):
        self._given_eta = value

    @max_iter.setter
    def max_iter(self, value: int):
        self._max_iter = value

    @given_path.setter
    def given_path(self, value: str):
        self._path = value

    @img_header.setter
    def img_header(self, value):
        self._img_header = value


    def __call__(self, iarray) -> np.ndarray:
        """
        Calls the AnatomicReconstructor Operator on the inputed array

        If saving_options are set then AnatomicReconstructor can pull from
        already constructed images
        """
        if self._normalize:
            self._ground_truth = normalize_matrix(iarray)
        else:
            self._ground_truth = iarray
        if self.saving:
            filename = self.search_image()
            if filename is not None:
                img_header = nib.as_closest_canonical(nib.load(filename))
                return img_header.get_fdata()
        reconstruction = self.run_reconstructor()
        self.save_image(reconstruction)
        return reconstruction
        
    
    def run_reconstructor(self):
        """
        Runs the reconstructor algorithm and masks the result to prevent the MSE being influenced by unnessary
        background pixels
        """
        downsampler = spl.AverageDownsampling(self._ground_truth.shape, self._downsampling_factor)
        downsampled = downsampler(self._ground_truth)
        gradient_op = sp.linop.FiniteDifference(self._ground_truth.shape)
        projection_op = proj.ProjectionOperator(gradient_op.oshape, self._anatomical_data, eta=self._given_eta)
        compose_op = sp.linop.Compose([projection_op, gradient_op])
        gproxy = sp.prox.L1Reg(compose_op.oshape, self._given_lambda)

        alg = sp.app.LinearLeastSquares(downsampler, downsampled, proxg=gproxy, G=compose_op, max_iter=self.max_iter)
        result = alg.run()
        masked_result = result * (self._ground_truth > 0.0001)
        return masked_result
    

    def save_image(self, img):
        """
        Saves reconstruction to a folder that can be read from later
        """
        recon_img = nib.Nifti1Image(img, self.img_header.affine, self.img_header.header)
        filename = f"lambda-{self._given_lambda}_eta-{self._given_eta}_iter-{self._max_iter}"
        if self.normalize:
            filename += "_norm"
        filename += ".nii"
        save_path = self._path.joinpath(filename)
        nib.save(recon_img, save_path)

    def search_image(self) -> Union[str, None]:
        """
        Checks if an image with the current settings has already been generated. 
        If so it returns the path otherwise it returns None
        """
        filename = f"lambda-{self._given_lambda}_eta-{self._given_eta}_iter-{self._max_iter}"
        if self.normalize:
            filename += "_norm"
        filename += ".nii"
        search_path = self._path.joinpath(filename)
        if search_path.exists():
            return search_path
        return None



def normalize_matrix(matrix):
    """
    normalizes inputed matrix so that all of it's values range from 0 - 1
    """
    m_max = matrix.max()
    return matrix/m_max
