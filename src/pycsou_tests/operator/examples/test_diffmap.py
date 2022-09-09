import itertools

import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.operator.linop as pycl
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class Sin(pyca.DiffMap):
    # f: \bR^{M} -> \bR^{M}
    #      x     -> sin(x)
    def __init__(self, M: int):
        super().__init__(shape=(M, M))
        self._lipschitz = 1
        self._diff_lipschitz = 1

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.sin(arr)
        return y

    def jacobian(self, arr):
        xp = pycu.get_array_module(arr)
        J = xp.cos(arr)
        return pycl.DiagonalOp(J)


class TestSin(conftest.DiffMapT):
    @pytest.fixture(
        params=itertools.product(
            (  # dim, op
                (4, Sin(M=4)),
                (1, Sin(M=1)),
            ),
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
        return (dim, dim)

    @pytest.fixture
    def data_apply(self, dim):
        A = np.linspace(0, np.pi / 2, dim)
        B = np.sin(A)
        return dict(
            in_=dict(arr=A),
            out=B,
        )

    @pytest.fixture
    def data_math_lipschitz(self, dim):
        N_test = 5
        return self._random_array((N_test, dim))

    @pytest.fixture
    def data_math_diff_lipschitz(self, dim):
        N_test = 5
        return self._random_array((N_test, dim))
