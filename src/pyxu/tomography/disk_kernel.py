import kernel
import numpy as np
import scipy

import pyxu.info.ptype as pyxt
import pyxu.util as pycu


class Disk(kernel.FiniteSupportPulse):
    def __init__(self, radius: float):
        super().__init__((1, 2))
        self._radius = radius
        assert radius >= 0

    def apply(self, arr: pyxt.NDArray) -> pyxt.NDArray:
        return (arr**2).sum(axis=-1) < self._radius**2

    def support(self, **kwargs) -> pyxt.Real:
        return self._radius if "support_disk_t" not in kwargs else float(kwargs["support_disk_t"])

    def applyF(self, arr: pyxt.NDArray) -> pyxt.NDArray:
        xp = pycu.get_array_module(arr)
        rho = xp.linalg.norm(arr, axis=-1)
        ret = (np.pi * (self._radius**2)) * np.ones(*arr.shape[:-1] if len(arr.shape) > 1 else 1)
        if not ret.shape:
            ret = xp.array([ret])
        tmp = rho[rho > 0]
        ret[rho > 0] = (self._radius / tmp) * scipy.special.j1(self._radius * 2 * np.pi * tmp)
        return ret

    def supportF(self, **kwargs) -> pyxt.Real:
        if "support_disk_f" in kwargs:
            return float(kwargs["support_disk_f"])
        return 20.4 / self._radius  # contains 99% of the energy, from photonics notes
