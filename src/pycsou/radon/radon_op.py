import numpy as np
import pyffs

import pycsou.abc as pyca
import pycsou.operator.linop as pyop
import pycsou.util.array_module as pycu
import pycsou.util.ptype as pyct
from pycsou.radon.finsupportfunc import FinSupFunc
from pycsou.util import view_as_complex


class RadonOp(pyca.LinOp):
    def __init__(self, shape: pyct.OpShape, delta, psi, n, t, extreme_shift, eps=5e-3, **kwargs):
        super().__init__(shape)
        self._dim = shape[-1]
        self._psi = psi
        self._delta = delta
        self._n = n
        self._t = t
        self._eps = eps

        xp = pycu.get_array_module(self._n)
        self._n = xp.array(list(zip([xp.cos(n), xp.sin(n)])))
        self._n_num = len(n)
        self._output_dim = self._n_num * self._t["num"]

        freq_support = self._psi.supportF(**kwargs)
        self._time_support = self._psi.support(**kwargs) + extreme_shift
        self._time_bandwidth_product = np.ceil(self._time_support * freq_support / 2.0)

        self._freqs = xp.arange(-self._time_bandwidth_product, self._time_bandwidth_product + 1)
        self._freqs = self._freqs.reshape(-1, 1) / self._time_support

        self._scaled_n = xp.kron(self._freqs, self._n)

        self._psi_applyF = self._psi.applyF(arr=self._scaled_n)

        adj_a_bar = np.exp(1j * 2.0 * np.pi * self._t["start"] / self._time_support)
        self._adj_w_bar = np.exp(1j * 2.0 * np.pi * (self._t["start"] - self._t["stop"]) / (self._t["num"] - 1))
        self._adj_a_vect = adj_a_bar ** xp.arange(-self._time_bandwidth_product, self._time_bandwidth_product + 1)
        self._adj_w_vect = (self._adj_w_bar ** (-self._time_bandwidth_product)) ** xp.arange(0, self._t["num"])

        self._apply_nufft = None
        self._adjoint_nufft = None

    def apply(self, alpha: pyct.NDArray) -> pyct.NDArray:
        """

        Parameters
        ----------
        alpha: (..., M)

        Returns
        -------
        Evaluating the Radon Transform with linear combination weights alpha at n, t, returns a vector of shape
        (..., Nn * Nt)

        """
        assert alpha.shape[-1] == self._dim
        arg = self.applyF(alpha) / self._time_support
        ret = pyffs.fs_interp(arg, self._time_support, self._t["start"], self._t["stop"], self._t["num"]).real
        return ret.reshape(-1, self._output_dim)

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        """

        Parameters
        ----------
        arr: (..., Nn * Nt)

        Returns
        -------
        Compute the adjoint of the Radon Transform at n, t, returns a vector of shape (..., Q)

        """
        assert arr.shape[-1] == self._output_dim
        arr = arr.reshape(-1, self._n_num, self._t)
        arg = pyffs.czt(self._adj_w_vect * arr, A=1, W=self._adj_w_bar, M=2 * self._time_bandwidth_product + 1, axis=-1)
        arg = self._adj_a_vect * arg
        return self._adjoint_nufft(arg)

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
        return (self._psi_applyF * view_as_complex(self._apply_nufft(alpha))).reshape(self._freqs.shape[0], -1).T

    @staticmethod
    def uniform(delta: dict, psi: FinSupFunc, n: dict, t: dict, eps=5e-3, **kwargs):
        """
        Instantiate a uniform RadonOp, i.e a RadonOp sampling the Radon transform of the sum of shifted copies of psi
        placed on a uniform 2D lattice over a uniform set of angles and offsets.

        Parameters
        ----------
        delta: dict {start: (2,), stop: (2,), num: (2,)}
        Centers of the pulses over a uniform 2D lattice. For each dimension d, the lattice goes through all points in
        2-dimensional space with d coordinates:
                    start[d] + k * (stop[d] - start[d])/(num[d] - 1)
        with k ranging from 0 to num[d] - 1.

        psi: FinSupFunc

        n: dict {start: (1,), stop: (1,), num: (1,)}
        Specification of uniform array of angles where the Radon transform is sampled. The array will contain all angles
        of the following values:
                    start +  k * (stop - start)/(num - 1)
        with k ranging from 0 to num - 1. start and stop will be assumed to be given in rad.

        t: dict {start: (1,), stop: (1,), num: (1,)}
        Specification of uniform array of offsets where the Radon transform is sampled. The array will contain all
        offsets of the following values:
                    start + k *  (stop - start)/(num - 1)

        eps: float
        Requested NUFFT accuracy

        Returns
        -------
        RadonOp of shape (n_num * t_num, delta_num[1] * delta_num[2])
        """
        xp = np if "array_module" not in kwargs else kwargs["array_module"]  # because nothing else to infer it from
        return _UniformRadonOp(
            (n["num"] * t["num"], np.prod(delta["num"]).squeeze()),
            (delta["num"][0], delta["num"][1]),
            psi,
            xp.linspace(**n),
            t,
            xp.max(np.array([[delta["start"][i], delta["stop"][i]] for i in [0, 1]]).flatten()),
            eps,
            **kwargs
        )

    @staticmethod
    def nonuniform(delta: pyct.NDArray, psi: FinSupFunc, n: pyct.NDArray, t: dict, eps=5e-3, **kwargs):
        """
        Instantiate a non-uniform RadonOp, i.e a RadonOp sampling the Radon transform of the sum of non uniformly
        placed shifted copies of psi over a non-uniform set of angles and a uniform set of offsets.

        Parameters
        ----------
        delta: (Q, 2)
        Centers of the pulses in 2-dimensional space.

        psi: FinSupFunc

        n: (Nn, )
        Angles where the Radon transform is sampled

        t: dict {start: (1,), stop: (1,), num: (1,)}
        Specification of uniform array of offsets where the Radon transform is sampled. The array will contain all
        offsets of the following values:
                    start + k *  (stop - start)/(num - 1)

        eps: float
        Requested NUFFT accuracy

        Returns
        -------
        RadonOp of shape (Nn * t_num, Q)
        """
        xp = pycu.get_array_module(n)
        return _NonUniformRadonOp(
            (len(n) * t["num"], len(delta)), delta, psi, n, t, xp.max(xp.abs(delta)), eps, **kwargs
        )


class _UniformRadonOp(RadonOp):
    def __init__(
        self,
        shape: pyct.OpShape,
        delta: pyct.OpShape,
        psi: FinSupFunc,
        n: pyct.NDArray,
        t: dict,
        extreme_shift: float,
        eps=5e-3,
        **kwargs
    ):
        super().__init__(shape, delta, psi, n, t, extreme_shift, eps, **kwargs)
        self._apply_nufft = pyop.nufft.NUFFT.type2(
            2 * np.pi * self._scaled_n, self._delta, isign=-1, real=True, eps=self._eps
        )
        self._adjoint_nufft = pyop.nufft.NUFFT.type1(
            2 * np.pi * self._scaled_n, self._delta, isign=1, real=True, eps=self._eps
        )


class _NonUniformRadonOp(RadonOp):
    def __init__(
        self,
        shape: pyct.OpShape,
        delta: pyct.NDArray,
        psi: FinSupFunc,
        n: pyct.NDArray,
        t: dict,
        extreme_shift: float,
        eps=5e-3,
        **kwargs
    ):
        super().__init__(shape, delta, psi, n, t, extreme_shift, eps, **kwargs)
        self._apply_nufft = pyop.nufft.NUFFT.type3(
            2 * np.pi * self._delta, self._scaled_n, isign=-1, real=True, eps=self._eps
        )
        self._adjoint_nufft = pyop.nufft.NUFFT.type3(
            self._scaled_n, 2 * np.pi * self._delta, isign=1, real=True, eps=self._eps
        )
