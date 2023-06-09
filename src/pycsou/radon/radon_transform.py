import numpy as np

import pycsou.operator.linop as pyop
import pycsou.util.array_module as pycu
import pycsou.util.ptype as pyct
from pycsou.util import view_as_complex, view_as_real


class FiniteSupport:
    def supportF(self, **kwargs) -> pyct.NDArray:
        return NotImplementedError

    def support(self, **kwargs) -> pyct.NDArray:
        return NotImplementedError

    def supportF_size(self, **kwargs) -> pyct.Real:
        tmp = self.supportF(**kwargs)
        return abs(max(tmp) - min(tmp))

    def support_size(self, **kwargs) -> pyct.Real:
        tmp = self.support(**kwargs)
        return abs(max(tmp) - min(tmp))

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplementedError

    def applyF(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplementedError


class RadonTransform:
    r"""
    Represent the 2D Radon Transform of a linear combination of shifted copies of a finite time and frequency support
    functional.
    """

    def __init__(
        self,
        psi: FiniteSupport,
        shifts: pyct.NDArray,
        n: pyct.NDArray,
        t: pyct.NDArray,
        lattice_shifts=False,
        lattice_t=False,
        **kwargs
    ):
        """

        Parameters
        ----------
        psi: D dimensional FiniteSupport function
        shifts: (M, 2) array or (int | (int, int)) if lattice_shifts is True
        n: (Nn, 2) array
        t: (Nt, ) array
        lattice_shifts: bool indicating whether shifts lie on a lattice
        """
        assert lattice_shifts or shifts.shape[1] == n.shape[1] == 2

        self._dim = shifts.shape[0] if not lattice_shifts else np.prod(shifts)
        self._psi = psi
        self._shifts = shifts
        self._n = n
        self._t = t
        self._lattice_shifts = lattice_shifts
        self._lattice_t = lattice_t

        xp = pycu.get_array_module(self._n)
        extreme_shift = xp.max(xp.abs(shifts))

        if "force_support" not in kwargs:
            freq_support = self._psi.supportF_size(**kwargs)
            self._time_support = self._psi.support_size(**kwargs) + extreme_shift
        else:
            self._time_support, freq_support = kwargs["force_support"]

        self._time_bandwidth_product = np.ceil(self._time_support * freq_support / 2.0)
        self._freqs = xp.arange(-self._time_bandwidth_product, self._time_bandwidth_product + 1).reshape(-1, 1)

    def apply(self, alpha: pyct.NDArray, eps=0) -> pyct.NDArray:
        """

        Parameters
        ----------
        alpha: (..., M)
        eps: Real: requested relative accuracy, must be non-negative

        Returns
        -------
        Evaluating the Radon Transform with linear combination weights alpha at n, t, returns a matrix of shape (Nn, Nt)

        """
        assert alpha.shape[-1] == self._dim

        arg = self.applyF(alpha, self._freqs / self._time_support, eps=eps) / self._time_support
        arg = view_as_real(np.ascontiguousarray(arg))

        return view_as_complex(
            pyop.NUFFT.type2(
                (2 * np.pi / self._time_support) * self._t,
                N=2 * self._time_bandwidth_product + 1,
                isign=1,
                real=False,
                eps=eps,
            ).apply(arg)
        ).real

    def applyF(self, alpha: pyct.NDArray, freqs: pyct.NDArray, eps=0) -> pyct.NDArray:
        """

        Parameters
        ----------
        alpha: (..., M)
        freqs: (..., 1)
        eps: Real: requested relative accuracy, must be non-negative

        Returns
        -------
        Evaluating the Fourier Transform of the Radon Transform with linear combination weights alpha along n at
        frequency vectors freqs

        """
        xp = pycu.get_array_module(self._n)
        scaled_n = xp.kron(freqs, self._n)
        if self._lattice_shifts:
            shifted_sum = pyop.nufft.NUFFT.type2(2 * np.pi * scaled_n, self._shifts, isign=-1, real=True, eps=eps)
        else:
            shifted_sum = pyop.nufft.NUFFT.type3(2 * np.pi * self._shifts, scaled_n, isign=-1, real=True, eps=eps)
        return (self._psi.applyF(arr=scaled_n) * view_as_complex(shifted_sum(alpha))).reshape(freqs.shape[0], -1).T
