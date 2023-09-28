import kernel
import numpy as np
import pyffs

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.util.array_module as pycu
from pyxu.operator import NUFFT
from pyxu.util import view_as_complex, view_as_real

FSP = kernel.FiniteSupportPulse

__all__ = [
    "XRayTransform",
]


class XRayTransform(pxa.LinOp):
    def __init__(
        self,
        arg_shape,
        ray_spec,
        origin=0,
        pitch=1,
        kernel=None,
        method: str = "ray-trace",
        **kwargs,
    ):
        # arg_shape: (D,)
        # ray_spec: dict(n=n_spec, t=t_spec)
        #     method = ray-trace
        #         n_spec = (N_ray, D) [determines transform backend]
        #         t_spec = (N_ray, D)
        #     method = fourier
        #         n_spec = (N_n, D)
        #         t_spec =
        #           * Uniform sampling
        #             (N_n, D, D-1) U_n matrices
        #             (D-1,) dict(start, stop, num)  [same as UniformSpread]
        #           * Non-Uniform sampling
        #             (N_n, D, D-1) U_n matrices
        #             (N_t, D-1) basis coefficients
        # origin: float | (D,)
        # pitch: float | (D,)
        #     Pixel-basis assumed
        # kernel: FSP | list[FSP]
        #     (..., 1) -> (..., 1) kernel (per (D-1,) dimensions)
        # method: ray-trace | fourier
        pass

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # (Override in sub-class.)
        # arr: (N_stack, N_1,...,N_D) float NP/CP
        # out: (N_stack, L)
        pass

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        # (Override in sub-class.)
        # arr: (N_stack, L) float NP/CP
        # out: (N_stack, N_1,...,N_D)
        pass


class _FourierXRayTransform(XRayTransform):
    def __init__(self, arg_shape, ray_spec, origin=0, pitch=1, kernel=None, **kwargs):
        # arg_shape: (D,)
        # ray_spec: dict(n=n_spec, t=t_spec)
        #     method = fourier
        #         n_spec = (N_n, D)
        #         t_spec =
        #           * Uniform sampling
        #             dict(start, stop, num)  [same as UniformSpread]
        #           * Non-Uniform sampling
        #             (N_t, D-1)
        #
        #             {k \in \bR^{D-1} | t = U_{n^{\perp}} k}
        #             Concretely, if n=(0, 1) [2D] or (0, 0, 1) [3D], then t_spec describes
        #             (x,) or (x, y) line offsets w.r.t 0^{D}.
        # origin: int | (D,)
        # pitch: int | (D,)
        #     Pixel-basis assumed
        # kernel: FSP | list[FSP]
        #     (..., 1) -> (..., 1) kernel (per (D-1,) dimensions)
        super().__init__(arg_shape, ray_spec, origin, pitch, kernel, **kwargs)
        self._argshape = arg_shape
        self._psi = kernel  # TODO account for multiple kernels (but how?)
        self._n = n = ray_spec["n"]
        self._t = ray_spec["t"]
        self._eps = 1e-6  # TODO ADD KWARG FOR NUFFT EPS

        xp = pycu.get_array_module(self._n)
        self._n_num = len(n)
        self._unif_t = self._t is dict
        self._t_num = self._t["num"] if self._unif_t else len(self._t)

        if self._t_num < 2:
            msg = "Need to sample the XRay transform at 2 offsets `t` at least."
            raise ValueError(msg)

        self._output_dim = self._n_num * self._t_num

        freq_support = self._psi.supportF(**kwargs)  # TODO instead, estimate numerically
        self._time_support = self._psi.support(**kwargs) + np.linalg.norm(
            pitch - origin
        )  # TODO account for pitch/origin not being vectors but just ints
        self._time_bandwidth_product = int(np.ceil(self._time_support * freq_support / 2.0))

        self._freqs = xp.arange(-self._time_bandwidth_product, self._time_bandwidth_product + 1)
        self._freqs = self._freqs / self._time_support

        self._scaled_n = xp.kron(self._n.T, self._freqs).T  # TODO projection to n^orthogonal

        self._psi_applyF = self._psi.applyF(arr=self._scaled_n) / self._time_support

        self._adj_a = xp.exp(-1j * 2.0 * np.pi * self._t["start"] / self._time_support)
        self._adj_w = xp.exp(
            1j * 2.0 * np.pi * (self._t["stop"] - self._t["start"]) / ((self._t_num - 1) * self._time_support)
        )
        self._adj_a_vect = self._adj_a ** (-xp.arange(-self._time_bandwidth_product, self._time_bandwidth_product + 1))
        self._adj_w_vect = (self._adj_w ** (-self._time_bandwidth_product)) ** xp.arange(0, self._t["num"])
        self._adj_psi_applyF = self._psi_applyF.reshape(self._freqs.shape[0], -1).T * self._adj_a_vect

        self._fourier_proj_nufft = NUFFT.type2(
            2 * np.pi * self._scaled_n, self._argshape, isign=-1, real=True, eps=self._eps
        )  # TODO recheck after projection to n^orthogonal

        if not self._unif_t:
            self._backproj = NUFFT.type2(
                (2 * np.pi / self._time_support) * self._t,
            )

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        # Low-Level interface. (Override in sub-class.)
        # arr: (N_stack, N_1,...,N_D) float NP/CP
        # out: (N_stack, L)
        assert arr.shape[1:] == self._argshape
        arg = self.applyF(arr)

        if self._unif_t:
            ret = pyffs.fs_interp(arg, self._time_support, self._t["start"], self._t["stop"], self._t["num"]).real
        else:
            ret = None

        if len(arr.shape) > 1:
            ret = ret.reshape(*arr.shape[:-1], self._n_num, self._t["num"])
            return ret.reshape(*arr.shape[:-1], self._output_dim)
        else:
            return ret.reshape(self._n_num, self._t["num"]).ravel()

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        # Low-Level interface. (Override in sub-class.)
        # arr: (N_stack, L) float NP/CP
        # out: (N_stack, N_1,...,N_D)

        assert arr.shape[-1] == self._output_dim
        arr = arr.reshape(*arr.shape[:-1], self._n_num, self._t["num"])

        arr = self._adj_w_vect * arr
        arg = pyffs.czt(arr, A=1.0, W=self._adj_w, M=2 * self._time_bandwidth_product + 1, axis=-1)
        arg = np.flip(arg, axis=-1)
        arg = self._adj_psi_applyF * arg
        arg = view_as_real(arg)
        ret = self._fourier_proj_nufft.adjoint(arg.ravel())

        ret = ret.squeeze()
        if not ret.shape:
            return np.array([ret])
        return ret

    def applyF(self, alpha: pxt.NDArray) -> pxt.NDArray:
        """

        Parameters
        ----------
        alpha: (..., Q)

        Returns
        -------
        Evaluating the Fourier Transform of the Radon Transform with linear combination weights alpha along n at
        regularly spaced frequency vectors

        """
        return (self._psi_applyF * view_as_complex(self._fourier_proj_nufft(alpha))).reshape(self._freqs.shape[0], -1).T
