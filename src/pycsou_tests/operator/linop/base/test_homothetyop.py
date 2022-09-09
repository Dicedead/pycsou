import itertools

import numpy as np
import pytest

import pycsou.operator as pyco
import pycsou.runtime as pycrt
import pycsou.util.deps as pycd
import pycsou_tests.operator.conftest as conftest


class TestHomothetyOp(conftest.PosDefOpT):
    @pytest.fixture(params=[-3.4, -1, 1, 2])  # cst=0 manually tested given different interface.
    def cst(self, request):
        return request.param

    @pytest.fixture
    def dim(self):
        return 5

    @pytest.fixture
    def data_shape(self, dim):
        return (dim, dim)

    @pytest.fixture(
        params=itertools.product(
            pycd.NDArrayInfo,
            pycrt.Width,
        )
    )
    def spec(self, cst, dim, request):
        op = pyco.HomothetyOp(cst=cst, dim=dim)
        return op, *request.param

    @pytest.fixture
    def data_apply(self, cst, dim):
        arr = np.arange(dim)
        out = cst * arr
        return dict(
            in_=dict(arr=arr),
            out=out,
        )

    def test_math_eig(self, _op_eig, cst):
        if cst < 0:
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_eig(_op_eig)

    def test_math_posdef(self, op, xp, width, cst):
        if cst < 0:
            pytest.skip("disabled since operator is not positive-definite.")
        else:
            super().test_math_posdef(op, xp, width)
