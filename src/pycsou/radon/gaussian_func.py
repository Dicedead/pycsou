import numpy as np
import scipy.integrate as spi
import scipy.special as sps
from finsupportfunc import FinSupFunc

import pycsou.util as pycu
import pycsou.util.ptype as pyct


class TruncatedGaussian(FinSupFunc):
    # f: [-1, 1] -> R
    #    x -> Exp[- (x / \sigma)^2 / 2]
    #
    # f^{F}: R -> R
    #    v -> \sqrt{2 \pi} \sigma
    #         Exp[-2 (\pi \sigma v)^2]
    #         Re[Erf[
    #             \frac{1}{\sqrt{2} \sigma}
    #             + j \sqrt{2} \pi \sigma v
    #         ]]

    def __init__(self, sigma: float):
        super().__init__((1, 2))
        self._sigma = float(sigma)
        assert 0 < self._sigma <= 1

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        w = arr / (xp.sqrt(2) * self._sigma)
        out = xp.exp(-xp.sum(w**2, axis=-1))
        out[xp.linalg.norm(arr, axis=-1) > 1] = 0
        return out

    def support(self, **kwargs) -> pyct.Real:
        return 1.0 if "support_gaussian_t" not in kwargs else float(kwargs["support_gaussian_t"])

    def applyF(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        w = np.sqrt(2) * np.pi * self._sigma * arr
        a = np.sqrt(2) * self._sigma

        b = xp.exp(-(w**2))
        c = sps.erf((1 / a) + 1j * w).real
        out = np.sqrt(np.pi) * a * b * c
        return out if len(w.shape) < 2 else np.prod(out, axis=-1)

    def supportF(self, **kwargs) -> pyct.Real:
        r"""
        Keyword arguments
        ----------
        eps: float
            Value :math:`\epsilon > 0` such that

            .. math::

               \sqrt{\int_{-s}^{s} |f^{F}(v)|^{2} dv}
               =
               (1 - \epsilon) \|f\|_{2}
        """
        if "support_gaussian_f" in kwargs:
            return float(kwargs["support_gaussian_f"])

        eps = 5e-3
        if "eps_gaussian_f" in kwargs:
            eps = float(kwargs["eps_gaussian_f"])

        tol = (1 - eps) ** 2

        def energy(v_max: float) -> float:
            # Estimate signal energy between [-v_max, v_max]
            E, err = spi.quadrature(
                func=lambda _: self.applyF(_) ** 2,
                a=-v_max,
                b=v_max,
            )
            return E

        E_tot = np.sqrt(np.pi) * self._sigma * sps.erf(1 / self._sigma)  # |f|^{2}

        # Coarse-grain search for a max bandwidth in v_step increments.
        tolerance_reached = False
        v_step = 1 / (2 * np.pi * self._sigma)
        v_max = float(v_step)
        while not tolerance_reached:
            v_max += v_step
            E = energy(v_max)
            tolerance_reached = E >= tol * E_tot

        # Fine-grained search for a max bandwidth in [v_max - v_step, v_max] region.
        v_fine = np.linspace(v_max - v_step, v_max, 100)
        E = np.array([energy(v) for v in v_fine])
        s = v_fine[E >= tol * E_tot].min()
        return s
