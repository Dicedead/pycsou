import numpy as np
import scipy.stats as ss

import pycsou.util.array_module as pycu
import pycsou.util.ptype as pyct
from pycsou.radon.radon_transform import FiniteSupport


class TruncatedGaussian(FiniteSupport):
    def __init__(self, dim: int, sigma: pyct.Real):
        self._dim = dim
        self._sigma = sigma

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        denom = (2 * np.pi * self._sigma**2) ** (self._dim / 2.0)
        return xp.exp(-(arr**2).sum(axis=-1) / (2 * (self._sigma**2))) / denom

    def applyF(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.exp((-2 * (np.pi * self._sigma) ** 2) * xp.sum(arr**2, axis=-1), dtype=np.complex_)

    def support(self, **kwargs) -> pyct.NDArray:
        q = 1e-5
        if "q" in kwargs:
            q = kwargs["q"]
        q = q / 2
        return np.array([ss.norm.isf(q=1 - q, scale=self._sigma), ss.norm.isf(q=q, scale=self._sigma)])

    def supportF(self, **kwargs) -> pyct.NDArray:
        return self.support(**kwargs)
