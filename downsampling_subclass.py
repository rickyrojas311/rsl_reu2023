"""
Subclass for the downsampling object in scipy and associated functions
"""
from __future__ import annotations

import sigpy as sp
from sigpy import backend
try:
    import cupy as xp
except ImportError:
    import numpy as xp
import numpy as np


class AverageDownsampling(sp.linop.Linop):
    """Downsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Downsampling factor.
        shift (None of tuple of ints): Shifts before down-sampling.
    """
    def __init__(self, ishape: tuple[int], factors: tuple[int]):
        self.factors = factors
        oshape = get_oshape(ishape, factors)

        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return downsample_average(input, self.factors)

    def _adjoint_linop(self):
        return AverageUpsampling(self.ishape, self.factors)    


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
        bucket_size = int(np.ceil(iarray.shape[dim]/downsampling_factor))
        # import ipdb; ipdb.set_trace()
        split_array = xp.array_split(iarray, bucket_size, axis=dim)
        for array in split_array:
            arrays.append(xp.mean(array, axis=dim))
        iarray = xp.stack(arrays, axis=dim)
    return iarray


def get_oshape(ishape, factors):
    """
    Calculates the output shape after downsampling.
    Factors must evenly divide ishape or else an error is raised
    """
    #If any of the modulo pairings between ishape and factors aren't 0 raise error
    if len(ishape) is not len(factors):
        raise ValueError(f"input shape is {ishape} with {len(ishape)} dimensions, factors {factors} are only set for {len(factors)} dimensions")
    if any(i % f for i, f in zip(ishape, factors)):
        raise ValueError(f"factors cause remainder, {ishape} should be evenly divisable by {factors}")
    return [i // f for i,f in zip(ishape, factors)]

class AverageUpsampling(sp.linop.Linop):
    """Upsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Upsampling factor.
        shift (None of tuple of ints): Shifts before up-sampling.

    """

    def __init__(self, oshape: tuple[int], factors: tuple[int]):
        self.factors = factors

        ishape = [i // f for i,f in zip(oshape, factors)]

        super().__init__(oshape, ishape)


    def _apply(self, input):
        device = backend.get_device(input)
        with device:
            return upsample_average(input, self.oshape, self.factors)

    def _adjoint_linop(self):
        return AverageDownsampling(self.oshape, self.factors)


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
        iarray = xp.repeat(iarray/multiplier, multiplier, axis=dim)
    slices = tuple(slice(None, dim) for dim in oshape)
    return iarray[slices]
