import itertools

import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class ScaledSum(pyca.LinFunc):
    # f: \bR^{M} -> \bR
    #      x     -> cumsum(x).sum()
    def __init__(self, N: int):
        super().__init__(shape=(1, N))
        self._lipschitz = np.sqrt(N * (N + 1) * (2 * N + 1) / 6)

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        return arr.cumsum(axis=-1).sum(axis=-1, keepdims=True)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        xp = pycu.get_array_module(arr)
        out = xp.zeros((*arr.shape[:-1], self.dim), dtype=arr.dtype)
        out[..., :] = xp.arange(self.dim, 0, -1, dtype=arr.dtype)
        out *= arr
        return out


class TestScaledSum(conftest.LinFuncT):
    @pytest.fixture(
        params=itertools.product(
            ((5, ScaledSum(N=5)),),  # dim, op
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def _spec(self, request):
        return request.param

    @pytest.fixture
    def spec(self, _spec):
        return _spec[0][1], _spec[1], _spec[2]

    @pytest.fixture
    def dim(self, _spec):
        return _spec[0][0]

    @pytest.fixture
    def data_shape(self, dim):
        return (1, dim)

    @pytest.fixture
    def data_apply(self, dim):
        x = self._random_array((dim,))
        y = x.cumsum().sum(keepdims=True)
        return dict(
            in_=dict(arr=x),
            out=y,
        )
