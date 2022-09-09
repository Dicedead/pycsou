import itertools

import numpy as np
import pytest

import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class Median(pyca.Func):
    # f: \bR^{M} -> \bR
    #      x     -> median(x)
    def __init__(self):
        super().__init__(shape=(1, None))
        self._lipschitz = np.inf

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        y = xp.median(arr, axis=-1, keepdims=True)
        return y


class TestMedian(conftest.FuncT):
    disable_test = frozenset(
        conftest.FuncT.disable_test
        | {
            "test_interface_asloss",  # does not make sense for Median().
        }
    )

    @pytest.fixture(
        params=itertools.product(
            (Median(),),
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, request):
        return request.param

    @pytest.fixture
    def data_shape(self):
        return (1, None)

    @pytest.fixture(
        params=[  # 2 test points
            dict(
                in_=dict(arr=np.arange(-5, 6)),
                out=np.array([0]),
            ),
            dict(
                in_=dict(arr=np.arange(200, 350)),
                out=np.array([274.5]),
            ),
        ]
    )
    def data_apply(self, request):
        return request.param

    @pytest.fixture
    def data_math_lipschitz(self):
        N_test, N_dim = 5, 3
        return self._random_array((N_test, N_dim))
