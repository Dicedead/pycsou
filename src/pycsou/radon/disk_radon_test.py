import numpy as np
import scipy

import pycsou.util as pycu
import pycsou.util.ptype as pyct
from pycsou.radon.radon_transform import FiniteSupportPulse


class Disk(FiniteSupportPulse):
    def __init__(self, radius: float):
        self._radius = radius
        assert radius >= 0

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return (arr**2).sum(axis=-1) < self._radius**2

    def support(self, **kwargs) -> pyct.Real:
        return self._radius if "support_disk_t" not in kwargs else float(kwargs["support_disk_t"])

    def applyF(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        rho = xp.linalg.norm(arr)
        return (self._radius / rho) * scipy.special.j1(self._radius * 2 * np.pi * rho)

    def supportF(self, **kwargs) -> pyct.Real:
        return float(kwargs["support_disk_f"])
        # TODO is of jinc <=> support of sinc by radial symmetry? (check, I think so)
        # TODO if so: using parseval, is there a way to estimate sinc (and thus jinc) energy from box function energy ?
