import numpy as np

import pycsou.operator.linop as pyop
import pycsou.util.array_module as pycu
import pycsou.util.ptype as pyct
from pycsou.util import view_as_complex, view_as_real


class FiniteSupportPulse:
    def supportF(self, **kwargs) -> pyct.Real:
        return NotImplementedError

    def support(self, **kwargs) -> pyct.Real:
        return NotImplementedError

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
        psi: FiniteSupportPulse,
        shifts: pyct.NDArray,
        n: pyct.NDArray,
        t: pyct.NDArray,
        lattice_shifts=False,
        eps=5e-3,
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
        eps: Real: requested relative accuracy, must be non-negative
        """
        assert lattice_shifts or shifts.shape[1] == n.shape[1] == 2

        self._dim = shifts.shape[0] if not lattice_shifts else np.prod(shifts)
        self._psi = psi
        self._shifts = shifts
        self._n = n
        self._t = t
        self._lattice_shifts = lattice_shifts
        self._eps = eps

        xp = pycu.get_array_module(self._n)
        extreme_shift = xp.max(xp.abs(shifts))

        freq_support = self._psi.supportF(**kwargs)
        self._time_support = self._psi.support(**kwargs) + extreme_shift
        self._time_bandwidth_product = np.ceil(self._time_support * freq_support / 2.0)

        self._freqs = xp.arange(-self._time_bandwidth_product, self._time_bandwidth_product + 1)
        self._freqs = self._freqs.reshape(-1, 1) / self._time_support

        scaled_n = xp.kron(self._freqs, self._n)
        if self._lattice_shifts:
            self._first_nufft = pyop.nufft.NUFFT.type2(
                2 * np.pi * scaled_n, self._shifts, isign=-1, real=True, eps=self._eps
            )
        else:
            self._first_nufft = pyop.nufft.NUFFT.type3(
                2 * np.pi * self._shifts, scaled_n, isign=-1, real=True, eps=self._eps
            )

        self._psi_applyF = self._psi.applyF(arr=scaled_n)

        self._second_nufft = pyop.NUFFT.type2(
            (2 * np.pi / self._time_support) * self._t,
            N=2 * self._time_bandwidth_product + 1,
            isign=1,
            real=False,
            eps=eps,
        )

    def apply(self, alpha: pyct.NDArray) -> pyct.NDArray:
        """

        Parameters
        ----------
        alpha: (..., M)

        Returns
        -------
        Evaluating the Radon Transform with linear combination weights alpha at n, t, returns a matrix of shape (Nn, Nt)

        """
        assert alpha.shape[-1] == self._dim

        arg = self.applyF(alpha) / self._time_support
        arg = view_as_real(np.ascontiguousarray(arg))
        return view_as_complex(self._second_nufft.apply(arg)).real

    def applyF(self, alpha: pyct.NDArray) -> pyct.NDArray:
        """

        Parameters
        ----------
        alpha: (..., M)

        Returns
        -------
        Evaluating the Fourier Transform of the Radon Transform with linear combination weights alpha along n at
        regularly spaced frequency vectors

        """
        return (self._psi_applyF * view_as_complex(self._first_nufft(alpha))).reshape(self._freqs.shape[0], -1).T
