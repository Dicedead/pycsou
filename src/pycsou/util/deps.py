import enum
import importlib.util
import types

import dask.array
import numpy
import scipy.sparse
import sparse

CUPY_ENABLED: bool = importlib.util.find_spec("cupy") is not None
if CUPY_ENABLED:
    try:
        import cupy
        import cupyx.scipy.sparse
        import cupyx.scipy.sparse.linalg
    except ImportError:
        # CuPy is installed, but GPU drivers probably missing.
        CUPY_ENABLED = False


@enum.unique
class NDArrayInfo(enum.Enum):
    """
    Supported dense array APIs.
    """

    NUMPY = enum.auto()
    DASK = enum.auto()
    CUPY = enum.auto()

    def type(self) -> type:
        if self.name == "NUMPY":
            return numpy.ndarray
        elif self.name == "DASK":
            return dask.array.core.Array
        elif self.name == "CUPY":
            return cupy.ndarray if CUPY_ENABLED else type(None)
        else:
            raise ValueError(f"No known array type for {self.name}.")

    @classmethod
    def from_obj(cls, obj) -> "NDArrayInfo":
        if obj is not None:
            for ndi in cls:
                if isinstance(obj, ndi.type()):
                    return ndi
        raise ValueError(f"No known array type to match {obj}.")

    def module(self, linalg: bool = False) -> types.ModuleType:
        if self.name == "NUMPY":
            xp = numpy
            xpl = xp.linalg
        elif self.name == "DASK":
            xp = dask.array
            xpl = xp.linalg
        elif self.name == "CUPY":
            xp = cupy if CUPY_ENABLED else None
            xpl = xp if (xp is None) else xp.linalg
        else:
            raise ValueError(f"No known module(s) for {self.name}.")
        return xpl if linalg else xp


@enum.unique
class SparseArrayInfo(enum.Enum):
    """
    Supported sparse array APIs.
    """

    SCIPY_SPARSE = enum.auto()
    PYDATA_SPARSE = enum.auto()
    CUPY_SPARSE = enum.auto()

    def type(self) -> type:
        if self.name == "SCIPY_SPARSE":
            # All `*matrix` and `*_array` classes descend from `spmatrix`.
            return scipy.sparse.spmatrix
        elif self.name == "PYDATA_SPARSE":
            return sparse.SparseArray
        elif self.name == "CUPY_SPARSE":
            return cupyx.scipy.sparse.spmatrix if CUPY_ENABLED else type(None)
        else:
            raise ValueError(f"No known array type for {self.name}.")

    @classmethod
    def from_obj(cls, obj) -> "SparseArrayInfo":
        if obj is not None:
            for sai in cls:
                if isinstance(obj, sai.type()):
                    return sai
        raise ValueError(f"No known array type to match {sai}.")

    def module(self, linalg: bool = False) -> types.ModuleType:
        if self.name == "SCIPY_SPARSE":
            xp = scipy.sparse
            xpl = xp.linalg
        elif self.name == "PYDATA_SPARSE":
            xp = sparse
            xpl = scipy.sparse.linalg
        elif self.name == "CUPY_SPARSE":
            xp = cupyx.scipy.sparse if CUPY_ENABLED else None
            xpl = xp if (xp is None) else cupyx.scipy.sparse.linalg
        else:
            raise ValueError(f"No known array module for {self.name}.")
        return xpl if linalg else xp


def supported_array_types():
    data = set()
    for ndi in NDArrayInfo:
        if (ndi != NDArrayInfo.CUPY) or CUPY_ENABLED:
            data.add(ndi.type())
    return tuple(data)


def supported_array_modules():
    data = set()
    for ndi in NDArrayInfo:
        if (ndi != NDArrayInfo.CUPY) or CUPY_ENABLED:
            data.add(ndi.module())
    return tuple(data)


def supported_sparse_types():
    data = set()
    for sai in SparseArrayInfo:
        if (sai != SparseArrayInfo.CUPY_SPARSE) or CUPY_ENABLED:
            data.add(sai.type())
    return tuple(data)


def supported_sparse_modules():
    data = set()
    for sai in SparseArrayInfo:
        if (sai != SparseArrayInfo.CUPY_SPARSE) or CUPY_ENABLED:
            data.add(sai.module())
    return tuple(data)


__all__ = [
    "CUPY_ENABLED",
    "NDArrayInfo",
    "SparseArrayInfo",
    "supported_array_types",
    "supported_array_modules",
    "supported_sparse_types",
    "supported_sparse_modules",
]
