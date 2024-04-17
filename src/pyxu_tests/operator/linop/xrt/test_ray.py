import itertools

import numpy as np
import pytest
import pyxu_xrt.operator as pxo

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.runtime as pxrt
import pyxu_tests.conftest as ct
import pyxu_tests.operator.conftest as conftest


class RayXRTMixin:
    @pytest.fixture
    def dim_shape(self, space_dim) -> pxt.NDArrayShape:
        if space_dim == 2:
            return (5, 6)
        else:
            return (5, 3, 4)

    @pytest.fixture
    def codim_shape(self, nt_spec) -> pxt.NDArrayShape:
        n_spec, _ = nt_spec
        return (len(n_spec),)

    # Fixtures (internal) -----------------------------------------------------
    @classmethod
    def _metric(
        cls,
        a: pxt.NDArray,
        b: pxt.NDArray,
        as_dtype: pxt.DType,
    ) -> bool:
        # Ray[W]XRT precision is always float32 internally.
        width = pxrt.Width.SINGLE
        return super()._metric(a, b, as_dtype=width.value)

    @pytest.fixture(params=[2, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def origin(self, space_dim) -> tuple[float]:
        # Volume origin
        orig = self._random_array((space_dim,))
        return tuple(orig)

    @pytest.fixture
    def pitch(self, space_dim) -> tuple[float]:
        # Voxel pitch
        pitch = abs(self._random_array((space_dim,))) + 1e-3
        return tuple(pitch)

    @pytest.fixture
    def nt_spec(self, dim_shape, origin, pitch) -> tuple[np.ndarray]:
        # To analytically test XRT correctness, we cast rays only along cardinal X/Y/Z directions, with one ray per
        # voxel side.

        D = len(dim_shape)
        n_spec = []
        t_spec = []
        for axis in range(D):
            # compute axes which are not projected
            dim = list(range(D))
            dim.pop(axis)

            # number of rays per dimension
            N_ray = np.array(dim_shape)[dim]

            n = np.zeros((*N_ray, D))
            n[..., axis] = 1
            n_spec.append(n.reshape(-1, D))

            t = np.zeros((*N_ray, D))
            _t = np.meshgrid(
                *[(np.arange(dim_shape[d]) + 0.5) * pitch[d] + origin[d] for d in dim],
                indexing="ij",
            )
            _t = np.stack(_t, axis=-1)
            t[..., dim] = _t
            t_spec.append(t.reshape(-1, D))

        n_spec = np.concatenate(n_spec, axis=0)
        t_spec = np.concatenate(t_spec, axis=0)
        return n_spec, t_spec


class TestRayXRT(RayXRTMixin, conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
                # pxd.NDArrayInfo.DASK,  # not yet supported
            ],
            pxrt.Width,
        )
    )
    def spec(
        self,
        dim_shape,
        origin,
        pitch,
        nt_spec,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        n_spec = ct.chunk_array(
            xp.array(nt_spec[0], dtype=width.value),
            complex_view=True,
            # `n_spec` is not a complex view, but its last axis cannot be chunked.
            # [See RayXRT() as to why.]
            # We emulate this by setting `complex_view=True`.
        )
        t_spec = ct.chunk_array(
            xp.array(nt_spec[1], dtype=width.value),
            complex_view=True,
            # `t_spec` is not a complex view, but its last axis cannot be chunked.
            # [See RayXRT() as to why.]
            # We emulate this by setting `complex_view=True`.
        )

        op = pxo.RayXRT(
            dim_shape=dim_shape,
            n_spec=n_spec,
            t_spec=t_spec,
            origin=origin,
            pitch=pitch,
            enable_warnings=False,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(
        self,
        dim_shape,
        pitch,
    ) -> conftest.DataLike:
        V = self._random_array(dim_shape)  # (N1,...,ND)

        D = len(dim_shape)
        P = []
        for axis in range(D):
            p = np.sum(V * pitch[axis], axis=axis)
            P.append(p.reshape(-1))
        P = np.concatenate(P, axis=0)

        return dict(
            in_=dict(arr=V),
            out=P,
        )


class TestRayWXRT(RayXRTMixin, conftest.LinOpT):
    @pytest.fixture(
        params=itertools.product(
            [
                pxd.NDArrayInfo.NUMPY,
                pxd.NDArrayInfo.CUPY,
                # pxd.NDArrayInfo.DASK,  # not yet supported
            ],
            pxrt.Width,
        )
    )
    def spec(
        self,
        dim_shape,
        origin,
        pitch,
        ntw_spec,
        request,
    ) -> tuple[pxt.OpT, pxd.NDArrayInfo, pxrt.Width]:
        ndi, width = request.param
        self._skip_if_unsupported(ndi)

        xp = ndi.module()
        n_spec = ct.chunk_array(
            xp.array(ntw_spec[0], dtype=width.value),
            complex_view=True,
            # `n_spec` is not a complex view, but its last axis cannot be chunked.
            # [See RayXRT() as to why.]
            # We emulate this by setting `complex_view=True`.
        )
        t_spec = ct.chunk_array(
            xp.array(ntw_spec[1], dtype=width.value),
            complex_view=True,
            # `t_spec` is not a complex view, but its last axis cannot be chunked.
            # [See RayXRT() as to why.]
            # We emulate this by setting `complex_view=True`.
        )
        w_spec = xp.array(ntw_spec[2], dtype=width.value)

        op = pxo.RayWXRT(
            dim_shape=dim_shape,
            n_spec=n_spec,
            t_spec=t_spec,
            w_spec=w_spec,
            origin=origin,
            pitch=pitch,
            enable_warnings=False,
        )
        return op, ndi, width

    @pytest.fixture
    def data_apply(
        self,
        dim_shape,
        pitch,
        ntw_spec,
    ) -> conftest.DataLike:
        V = self._random_array(dim_shape)  # (N1,...,ND)
        w = ntw_spec[2]  # weights

        D = len(dim_shape)
        P = []
        for axis in range(D):
            # Compute accumulated attenuation
            pad_width = [(0, 0)] * D
            pad_width[axis] = (1, 0)
            selector = [slice(None)] * D
            selector[axis] = slice(0, -1)
            _w = np.pad(w, pad_width)[tuple(selector)]

            A = np.exp(-pitch[axis] * np.cumsum(_w, axis=axis))
            B = np.where(
                np.isclose(w, 0),
                pitch[axis],
                (1 - np.exp(-w * pitch[axis])) / w,
            )
            p = np.sum(V * A * B, axis=axis)
            P.append(p.reshape(-1))
        P = np.concatenate(P, axis=0)

        return dict(
            in_=dict(arr=V),
            out=P,
        )

    # Fixtures (internal) -----------------------------------------------------
    @pytest.fixture
    def ntw_spec(self, dim_shape, nt_spec) -> tuple[np.ndarray]:
        # We use the same setup as RayXRT, just add weighting.
        n_spec, t_spec = nt_spec

        # To avoid numerical inaccuracies in computing the ground-truth [due to use of np.exp()],
        # we limit the range of valid `w`.
        rng = np.random.default_rng()
        w_spec = np.linspace(0.5, 1, np.prod(dim_shape), endpoint=True)
        w_spec *= rng.choice([-1, 1], size=w_spec.shape)
        w_spec = w_spec.reshape(dim_shape)

        return n_spec, t_spec, w_spec
