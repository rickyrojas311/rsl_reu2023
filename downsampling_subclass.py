"""
Subclass for the downsampling object in scipy
"""

import scipy

class averageDownsampling():
    """Downsampling linear operator.

    Args:
        ishape (tuple of ints): Input shape.
        factor (tuple of ints): Downsampling factor.
        shift (None of tuple of ints): Shifts before down-sampling.
    """

    def __init__(self, ishape, factors, shift=None):
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
            return util.downsample(input, self.factors, shift=self.shift)

    def _adjoint_linop(self):
        return Upsample(self.ishape, self.factors, shift=self.shift)