"""
Subclass for the downsampling object in scipy and associated functions
"""

import sigpy as sp
from sigpy import backend
import numpy as np

class AverageDownsampling(sp.linop.Linop):
    """Downsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Downsampling factor.
        shift (None of tuple of ints): Shifts before down-sampling.
    """

    def __init__(self, ishape: tuple[int], factors: tuple[int], shift=None):
        self.factors = factors

        if shift is None:
            shift = [0] * len(ishape)

        self.shift = shift
        oshape = [
            ((i - s + f - 1) // f) for i, f, s in zip(ishape, factors, shift)
        ]
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return downsample_average(input, self.factors)

    def _adjoint_linop(self):
        return AverageUpsampling(self.ishape, self.factors, shift=self.shift)
    


def downsample_average(iarray, factor: tuple[int]):
    """
    Downsamples inputed target array by the inputed factor tuple by 
    averaging consecutive values together.
    Reads the downsampling factor in the factor tuple for each dimension
    then splits the array into buckets and averages all the values in the buckets
    together

    @parameter
    factor downsamples by given factor in direction given in the tuple
    """
    for dim in range(iarray.ndim):
        if dim < len(factor):
            downsampling_factor = factor[dim]
        else:
            downsampling_factor = 1
        arrays = []
        bucket_size = np.ceil(iarray.shape[dim]/downsampling_factor)
        split_array = np.array_split(iarray, bucket_size, axis=dim)
        for array in split_array:
            arrays.append(np.mean(array, axis=dim))
        iarray = np.stack(arrays, axis=dim)
    return iarray


class AverageUpsampling(sp.linop.Linop):
    """Upsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Upsampling factor.
        shift (None of tuple of ints): Shifts before up-sampling.

    """

    def __init__(self, oshape: tuple[int], factors: tuple[int], shift=None):
        self.factors = factors

        if shift is None:
            shift = [0] * len(oshape)

        self.shift = shift
        ishape = [
            ((i - s + f - 1) // f) for i, f, s in zip(oshape, factors, shift)
        ]

        super().__init__(oshape, ishape)


    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return upsample_average(input, self.oshape, self.factors)

    def _adjoint_linop(self):
        return AverageDownsampling(self.oshape, self.factors, shift=self.shift)


def upsample_average(iarray, oshape, factors):
    """
    Upsamples the input array by the given factors into the given output shape
    """
    for dim in range(iarray.ndim):
        #Extracts the multiplier from factors with error correction
        if dim < len(factors):
            multiplier = factors[dim]
        else:
            multiplier = 1
        iarray = np.repeat(iarray/multiplier, multiplier, axis=dim)
    slices = tuple(slice(None, dim) for dim in oshape)
    return iarray[slices]

if __name__ == "__main__":
    np.random.seed(100)
    print(A := np.arange(12))
    print(upsample_average(A, (36,), (3,)))
