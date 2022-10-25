import math
import numbers
import types
import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.abc.operator as pyco
import pycsou.math.stencil as pycstencil
import pycsou.runtime as pycrt
import pycsou.util as pycu
import pycsou.util.deps as pycd
import pycsou.util.ptype as pyct
import pycsou.util.warning as pycuw

__all__ = ["IdentityOp", "NullOp", "NullFunc", "HomothetyOp", "DiagonalOp", "Stencil", "SumOp"]


class IdentityOp(pyca.OrthProjOp):
    def __init__(self, dim: pyct.Integer):
        super().__init__(shape=(dim, dim))

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return pycu.read_only(arr)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return pycu.read_only(arr)

    def svdvals(self, **kwargs) -> pyct.NDArray:
        return pyca.UnitOp.svdvals(self, **kwargs)

    def eigvals(self, **kwargs) -> pyct.NDArray:
        return pyca.UnitOp.svdvals(self, **kwargs)

    def asarray(self, **kwargs) -> pyct.NDArray:
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())
        A = xp.eye(N=self.dim, dtype=dtype)
        return A

    @pycrt.enforce_precision(i="arr")
    def pinv(self, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
        out = arr.copy()
        out /= 1 + kwargs.pop("damp", 0)
        return out

    def dagger(self, **kwargs) -> pyct.OpT:
        cst = 1 / (1 + kwargs.pop("damp", 0))
        op = HomothetyOp(cst=cst, dim=self.dim)
        return op

    def trace(self, **kwargs) -> pyct.Real:
        return float(self.dim)


class NullOp(pyca.LinOp):
    """
    Null operator.

    This operator maps any input vector on the null vector.
    """

    def __init__(self, shape: pyct.Shape):
        super().__init__(shape=shape)
        self._lipschitz = 0

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.codim),
        )

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        xp = pycu.get_array_module(arr)
        return xp.broadcast_to(
            xp.array(0, arr.dtype),
            (*arr.shape[:-1], self.dim),
        )

    def svdvals(self, **kwargs) -> pyct.NDArray:
        N = pycd.NDArrayInfo
        xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
        D = xp.zeros(kwargs.pop("k"), dtype=pycrt.getPrecision().value)
        return D

    def gram(self) -> pyct.OpT:
        op = NullOp(shape=(self.dim, self.dim))
        return op.asop(pyca.SelfAdjointOp).squeeze()

    def cogram(self) -> pyct.OpT:
        op = NullOp(shape=(self.codim, self.codim))
        return op.asop(pyca.SelfAdjointOp).squeeze()

    def asarray(self, **kwargs) -> pyct.NDArray:
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())
        A = xp.zeros(self.shape, dtype=dtype)
        return A

    def trace(self, **kwargs) -> pyct.Real:
        return float(0)


def NullFunc(dim: pyct.Integer) -> pyct.OpT:
    """
    Null functional.

    This functional maps any input vector on the null scalar.
    """
    op = NullOp(shape=(1, dim)).squeeze()
    op._name = "NullFunc"
    return op


def HomothetyOp(cst: pyct.Real, dim: pyct.Integer) -> pyct.OpT:
    """
    Scaling operator.

    Parameters
    ----------
    cst: pyct.Real
        Scaling factor.
    dim: pyct.Integer
        Dimension of the domain.

    Returns
    -------
    op: pyct.OpT
        (dim, dim) scaling operator.

    Notes
    -----
    This operator is not defined in terms of :py:func:`~pycsou.operator.linop.DiagonalOp` since it
    is array-backend-agnostic.
    """
    assert isinstance(cst, pyct.Real), f"cst: expected real, got {cst}."

    if np.isclose(cst, 0):
        op = NullOp(shape=(dim, dim))
    elif np.isclose(cst, 1):
        op = IdentityOp(dim=dim)
    else:  # build PosDef or SelfAdjointOp

        @pycrt.enforce_precision(i="arr")
        def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
            out = arr.copy()
            out *= _._cst
            return out

        def op_svdvals(_, **kwargs) -> pyct.NDArray:
            N = pycd.NDArrayInfo
            xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
            D = xp.full(
                shape=kwargs.pop("k"),
                fill_value=abs(_._cst),
                dtype=pycrt.getPrecision().value,
            )
            return D

        def op_eigvals(_, **kwargs) -> pyct.NDArray:
            D = _.svdvals(**kwargs)
            D *= np.sign(_._cst)
            return D

        @pycrt.enforce_precision(i="arr")
        def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
            out = arr.copy()
            scale = _._cst / (_._cst**2 + kwargs.pop("damp", 0))
            out *= scale
            return out

        def op_dagger(_, **kwargs) -> pyct.OpT:
            scale = _._cst / (_._cst**2 + kwargs.pop("damp", 0))
            op = HomothetyOp(cst=scale, dim=_.dim)
            return op

        def op_gram(_):
            return HomothetyOp(cst=_._cst**2, dim=_.dim)

        def op_trace(_, **kwargs):
            out = _._cst * _.codim
            return float(out)

        klass = pyca.PosDefOp if (cst > 0) else pyca.SelfAdjointOp
        op = klass(shape=(dim, dim))
        op._cst = cst
        op._lipschitz = abs(cst)
        op.apply = types.MethodType(op_apply, op)
        op.svdvals = types.MethodType(op_svdvals, op)
        op.eigvals = types.MethodType(op_eigvals, op)
        op.pinv = types.MethodType(op_pinv, op)
        op.dagger = types.MethodType(op_dagger, op)
        op.gram = types.MethodType(op_gram, op)
        op.cogram = op.gram
        op.trace = types.MethodType(op_trace, op)
        op._name = "HomothetyOp"

    return op.squeeze()


def DiagonalOp(
    vec: pyct.NDArray,
    enable_warnings: bool = True,
) -> pyct.OpT:
    r"""
    Diagonal linear operator :math:`L: \mathbf{x} \to \text{diag}(\mathbf{v}) \mathbf{x}`.

    Notes
    -----
    :py:func:`~pycsou.operator.linop.base.DiagonalOp` instances are **not arraymodule-agnostic**:
    they will only work with NDArrays belonging to the same array module as ``vec``.
    Moreover, inner computations may cast input arrays when the precision of ``vec`` does not match
    the user-requested precision.
    If such a situation occurs, a warning is raised.

    Parameters
    ----------
    vec: pyct.NDArray
        (N,) diagonal scale factors.
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.
    """
    assert len(vec) == np.prod(vec.shape), f"vec: {vec.shape} is not a DiagonalOp generator."
    if (dim := vec.size) == 1:  # Module-agnostic
        return HomothetyOp(cst=float(vec), dim=1)
    else:
        xp = pycu.get_array_module(vec)
        if pycu.compute(xp.allclose(vec, 0)):
            op = NullOp(shape=(dim, dim))
        elif pycu.compute(xp.allclose(vec, 1)):
            op = IdentityOp(dim=dim)
        else:  # build PosDef or SelfAdjointOp

            @pycrt.enforce_precision(i="arr")
            def op_apply(_, arr):
                if (_._vec.dtype != arr.dtype) and _._enable_warnings:
                    msg = "Computation may not be performed at the requested precision."
                    warnings.warn(msg, pycuw.PrecisionWarning)
                out = arr.copy()
                out *= _._vec
                return out

            def op_asarray(_, **kwargs) -> pyct.NDArray:
                N = pycd.NDArrayInfo
                dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
                xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())

                v = pycu.compute(_._vec.astype(dtype=dtype, copy=False))
                v = pycu.to_NUMPY(v)
                A = xp.diag(v)
                return A

            def op_gram(_):
                return DiagonalOp(
                    vec=_._vec**2,
                    enable_warnings=_._enable_warnings,
                )

            def op_svdvals(_, **kwargs):
                k = kwargs.pop("k")
                which = kwargs.pop("which", "LM")
                N = pycd.NDArrayInfo
                xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
                D = xp.abs(pycu.compute(_._vec))
                D = D[xp.argsort(D)]
                D = D.astype(pycrt.getPrecision().value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            def op_eigvals(_, **kwargs):
                k = kwargs.pop("k")
                which = kwargs.pop("which", "LM")
                N = pycd.NDArrayInfo
                xp = {True: N.CUPY, False: N.NUMPY}[kwargs.pop("gpu", False)].module()
                D = pycu.compute(_._vec)
                D = D[xp.argsort(xp.abs(D))]
                D = D.astype(pycrt.getPrecision().value, copy=False)
                return D[:k] if (which == "SM") else D[-k:]

            @pycrt.enforce_precision(i="arr")
            def op_pinv(_, arr: pyct.NDArray, **kwargs) -> pyct.NDArray:
                damp = kwargs.pop("damp", 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                out = arr.copy()
                out *= scale
                return out

            def op_dagger(_, **kwargs) -> pyct.OpT:
                damp = kwargs.pop("damp", 0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scale = _._vec / (_._vec**2 + damp)
                    scale[xp.isnan(scale)] = 0
                return DiagonalOp(
                    vec=scale,
                    enable_warnings=_._enable_warnings,
                )

            def op_trace(_, **kwargs):
                return float(_._vec.sum())

            klass = pyca.PosDefOp if pycu.compute(xp.all(vec > 0)) else pyca.SelfAdjointOp
            op = klass(shape=(dim, dim))
            op._vec = vec
            op._enable_warnings = bool(enable_warnings)
            op._lipschitz = pycu.compute(xp.abs(vec).max())
            op.apply = types.MethodType(op_apply, op)
            op.asarray = types.MethodType(op_asarray, op)
            op.gram = types.MethodType(op_gram, op)
            op.cogram = op.gram
            op.svdvals = types.MethodType(op_svdvals, op)
            op.eigvals = types.MethodType(op_eigvals, op)
            op.pinv = types.MethodType(op_pinv, op)
            op.dagger = types.MethodType(op_dagger, op)
            op.trace = types.MethodType(op_trace, op)
            op._name = "DiagonalOp"

        return op.squeeze()


def _ExplicitLinOp(
    cls: pyct.OpC,
    mat: typ.Union[pyct.NDArray, pyct.SparseArray],
    enable_warnings: bool = True,
) -> pyct.OpT:
    r"""
    Build a linear operator from its matrix representation.

    Given a matrix :math:`\mathbf{A}\in\mathbb{R}^{M\times N}`, the *explicit linear operator*
    associated to :math:`\mathbf{A}` is defined as

    .. math::

       f_\mathbf{A}(\mathbf{x})
       =
       \mathbf{A}\mathbf{x},
       \qquad
       \forall \mathbf{x}\in\mathbb{R}^N,

    with adjoint given by:

    .. math::

       f^\ast_\mathbf{A}(\mathbf{z})
       =
       \mathbf{A}^T\mathbf{z},
       \qquad
       \forall \mathbf{z}\in\mathbb{R}^M.

    Parameters
    ----------
    cls: pyct.OpC
        LinOp sub-class to instantiate.
    mat: pyct.NDArray | pyct.SparseArray
        (M, N) matrix generator.
        The input array can be *dense* or *sparse*.
        Accepted sparse arrays are:

        * CPU: COO/CSC/CSR/BSR/GCXS
        * GPU: COO/CSC/CSR
    enable_warnings: bool
        If ``True``, emit a warning in case of precision mis-match issues.

    Notes
    -----
    * :py:class:`~pycsou.operator.linop.base._ExplicitLinOp` instances are **not
      arraymodule-agnostic**: they will only work with NDArrays belonging to the same (dense) array
      module as ``mat``.
      Moreover, inner computations may cast input arrays when the precision of ``mat`` does not
      match the user-requested precision.
      If such a situation occurs, a warning is raised.

    * The matrix provided in ``__init__()`` is used as-is and can be accessed via ``.mat``.
    """

    def _standard_form(A):
        fail_dense = False
        try:
            pycd.NDArrayInfo.from_obj(A)
        except:
            fail_dense = True

        fail_sparse = False
        try:
            pycd.SparseArrayInfo.from_obj(A)
        except:
            fail_sparse = True

        if fail_dense and fail_sparse:
            raise ValueError("mat: format could not be inferred.")
        else:
            return A

    def _matmat(A, b, warn: bool = True) -> pyct.NDArray:
        # A: (M, N) dense/sparse
        # b: (..., N) dense
        # out: (..., M) dense
        if (A.dtype != b.dtype) and warn:
            msg = "Computation may not be performed at the requested precision."
            warnings.warn(msg, pycuw.PrecisionWarning)

        M, N = A.shape
        sh_out = (*b.shape[:-1], M)
        b = b.reshape((-1, N)).T  # (N, (...).prod)
        out = A.dot(b)  # (M, (...).prod)
        return out.T.reshape(sh_out)

    @pycrt.enforce_precision(i="arr")
    def op_apply(_, arr: pyct.NDArray) -> pyct.NDArray:
        return _matmat(_.mat, arr, warn=_._enable_warnings)

    @pycrt.enforce_precision(i="arr")
    def op_adjoint(_, arr: pyct.NDArray) -> pyct.NDArray:
        return _matmat(_.mat.T, arr, warn=_._enable_warnings)

    def op_asarray(_, **kwargs) -> pyct.NDArray:
        N = pycd.NDArrayInfo
        S = pycd.SparseArrayInfo
        dtype = kwargs.pop("dtype", pycrt.getPrecision().value)
        xp = kwargs.pop("xp", pycd.NDArrayInfo.NUMPY.module())

        try:  # Sparse arrays
            info = S.from_obj(_.mat)
            if info in (S.SCIPY_SPARSE, S.CUPY_SPARSE):
                f = lambda _: _.toarray()
            elif info == S.PYDATA_SPARSE:
                f = lambda _: _.todense()
            A = f(_.mat.astype(dtype))  # `copy` field not ubiquitous
        except:  # Dense arrays
            info = N.from_obj(_.mat)
            A = pycu.compute(_.mat.astype(dtype, copy=False))
        finally:
            A = pycu.to_NUMPY(A)

        return xp.array(A, dtype=dtype)

    def op_trace(_, **kwargs) -> pyct.Real:
        if _.dim != _.codim:
            raise NotImplementedError
        else:
            try:
                tr = _.mat.trace()
            except:
                # .trace() missing for [PYDATA,CUPY]_SPARSE API.
                S = pycd.SparseArrayInfo
                info = S.from_obj(_.mat)
                if info == S.PYDATA_SPARSE:
                    # use `sparse.diagonal().sum()`, but array must be COO.
                    try:
                        A = _.mat.tocoo()  # GCXS inputs
                    except:
                        A = _.mat  # COO inputs
                    finally:
                        tr = info.module().diagonal(A).sum()
                elif info == S.CUPY_SPARSE:
                    tr = _.mat.diagonal().sum()
                else:
                    raise ValueError(f"Unknown sparse format {_.mat}.")
            return float(tr)

    def op_lipschitz(_, **kwargs) -> pyct.Real:
        # We want to piggy-back onto Lin[Op,Func].lipschitz() to compute the Lipschitz constant L.
        # Problem: LinOp.lipschitz() relies on svdvals() or hutchpp() to compute L, and they take
        # different parameters to do computations on the GPU.
        # Solution:
        # * we add the relevant kwargs before calling the LinOp.lipschitz() + drop all unrecognized
        #   kwargs there as needed.
        # * similarly for LinFunc.lipschitz().
        N = pycd.NDArrayInfo
        S = pycd.SparseArrayInfo

        try:  # Dense arrays
            info = N.from_obj(_.mat)
            kwargs.update(
                xp=info.module(),
                gpu=info == N.CUPY,
            )
        except:  # Sparse arrays
            info = S.from_obj(_.mat)
            gpu = info == S.CUPY_SPARSE
            kwargs.update(
                xp=N.CUPY.module() if gpu else N.NUMPY.module(),
                gpu=gpu,
            )

        if _.codim == 1:
            L = pyca.LinFunc.lipschitz(_, **kwargs)
        else:
            L = _.__class__.lipschitz(_, **kwargs)
        return L

    op = cls(shape=mat.shape)
    op.mat = _standard_form(mat)
    op._enable_warnings = bool(enable_warnings)
    op.apply = types.MethodType(op_apply, op)
    op.adjoint = types.MethodType(op_adjoint, op)
    op.asarray = types.MethodType(op_asarray, op)
    op.lipschitz = types.MethodType(op_lipschitz, op)
    op.trace = types.MethodType(op_trace, op)
    op._name = "_ExplicitLinOp"
    return op


class TrimOp(pyco.LinOp):
    def __init__(self, arg_shape, widths):
        r"""
        Parameters
        ----------
        shape: int | tuple(int, int)
            Shape of the operator.
        widths: tuple( tuple(int, int), ...)
            Tuple containing one tuple per dimension of the input array. Each inner tuple contains the number of
            elements desired (width) to trim from the two extremes of that dimension.
        """
        assert (len(arg_shape) + 1) == len(widths), "`arg_shape` and `widths` must have the same number of elements"
        self.arg_shape = arg_shape
        self.widths = widths
        out_shape = [s - np.sum(widths[i + 1]) for i, s in enumerate(arg_shape)]
        super(TrimOp, self).__init__(shape=(np.prod(arg_shape), np.prod(out_shape)))

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Trim sides from an array.
        """
        slices = []
        for (start, end) in self.widths:
            end = None if end == 0 else -end
            slices.append(slice(start, end))
        return arr.reshape(-1, *self.arg_shape)[tuple(slices)]

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Pad input according to the widths.
        """
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(-1, *self.arg_shape)
        for i in range(1, len(self.widths)):

            _pad_width = tuple([(0, 0) if i != j else self.widths[i] for j in range(len(self.widths))])
            arr = xp.pad(array=arr, pad_width=_pad_width, mode="constant", cval=0.0)
        return arr


class Stencil(pyco.SquareOp):
    r"""
    Base class for NDArray computing functions that operate only on a local region of the NDArray through a
    multi-dimensional kernel, namely through correlation and convolution.

    This class leverages the :py:func:`numba.stencil` decorator, which allowing to JIT (Just-In-Time) compile these
    functions to run more quickly.


    Examples
    --------

    The following example creates a Stencil operator based on a 2-dimensional kernel. It shows how to perform correlation
    and convolution in CPU, GPU (Cupy) and distributed across different cores (Dask).


    .. code-block:: python3

        from pycsou.operator.linop.base import Stencil
        import numpy as np
        import cupy as cp
        import dask.array as da

        nsamples = 2
        data_shape = (500, 100)
        da_blocks = (50, 10)

        # Numpy
        data = np.ones((nsamples, *data_shape)).reshape(nsamples, -1)
        # Cupy
        data_cu = cp.ones((nsamples, *data_shape)).reshape(nsamples, -1)
        # Dask
        data_da = da.from_array(data, chunks=da_blocks).reshape(nsamples, -1)

        kernel = np.array([[0.5, 0, 0.5], [0, 0, 0], [0.5, 0, 0.5]])
        center = np.array([1, 0])

        stencil = Stencil(stencil_coefs=kernel, center=center, arg_shape=data_shape, boundary=0.)
        stencil_cu = Stencil(stencil_coefs=cp.asarray(kernel), center=center, arg_shape=data_shape, boundary=0.)

        # Correlate images with kernels
        out = stencil(data).reshape(nsamples, *data_shape)
        out_da = stencil(data_da).reshape(nsamples, *data_shape).compute()
        out_cu = stencil_cu(data_cu).reshape(nsamples, *data_shape).get()

        # Convolve images with kernels
        out_adj = stencil.adjoint(data).reshape(nsamples, *data_shape)
        out_da_adj = stencil.adjoint(data_da).reshape(nsamples, *data_shape).compute()
        out_cu_adj = stencil_cu.adjoint(data_cu).reshape(nsamples, *data_shape).get()

    Remark 1
    --------
    The :py:class:`~pycsou.operator.linop.base.Stencil` allows to perform both correlation and convolution. By default,
    the ``apply`` method will perform **correlation** of the input array with the given kernel / stencil, whereas the
    ``adjoint`` method will perform **convolution**.

    Remark 2
    --------
    When instantiated with a multi-dimensional kernel, the :py:class:`~pycsou.operator.linop.base.Stencil` performs
    convolution and correlation operations as non-separable filters. When possible, the user can decide whether to
    separate the filtering operation by composing different stencils for different axis to accelerate performance. This
    approach is not guaranteed to improve performance due to the repeated copying of arrays due internal padding
    operations.

    Remark 3
    --------
    There are five padding mode supported: ‘reflect’, ‘periodic’, ‘nearest’, ‘none’ (zero padding), or 'cval'
    (constant value). If a str or a real value are given, the same padding is applied to all dimensions in the domain of
    the stencil computation. To specify different padding modes for each axis, use a tuple or a dict. The padding depth
    is automatically set to guarantee that the kernel always find support when centered in the extremes of the input
    array.

    Remark 4
    --------
    Stencil computations on Dask arrays are performed with :py:func:`~Dask.array.map_overlap`. Please note that in that
    scnario, if an asymmetric kernel is used to instantiate the Stencil class, a new symmetric kernel (padded with
    zeros) will be used instead (except for the padding option 'none', which accepts non-symmetric kernels).

    Remark 5
    --------
    By default, for GPU computing, the ``threadsperblock`` argument is set according to the following criteria:

    .. math::

        \prod_{i=0}^{D-1} t_{i} \leq c

    where :math:`t_{i}` is the number of threads per block in dimension :math:`i`, :math:`D` is the number of dimensions
    of the kernel, and  :math:`c=1024` is the `limit number of threads per block in current GPUs
    <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>`_.

    Because arrays are stored in row-major order, a larger number of threds per block in the last axis of the Cupy array
    benefits the spatial locality in memory caching. For this reason ``threadsperblock`` is set to the maximum number in
    the last axis, and to the minimum possible (respecting the kernel shape) in the other axes.

    .. math::

        t_{i} = 2^{j} \geq k_{i}, s.t., 2^{j-1} < k_{i} \quad \textrm{for} \quad i\in[0, \dots, D-2],

    where :math:`k_{i}` is the size of the kernel in dimension :math:`i`, and:

    .. math::

        t_{D-1} = \frac{1024}{\prod_{i=1}^{D-2}}

    """

    def __init__(
        self,
        stencil_coefs: pyct.NDArray,
        center: pyct.NDArray,
        arg_shape: pyct.Shape,
        boundary: typ.Optional[typ.Union[pyct.Real, str, tuple, dict]] = None,
    ):
        r"""

        Parameters
        ----------
        stencil_coefs: NDArray
            Stencil coefficients. Must have the same number of dimension as the input array's arg_shape (i.e., without the
            stacking dimension).
        center: NDArray
            Index of the kernel's center. Must be a 1-dimensional array with one element per dimension in ``stencil_coefs``.
        arg_shape: tuple
            Shape of the input array.
        boundary: real, str, tuple, or dict , keyword only.
            How to handle the boundaries. Values include ‘reflect’, ‘periodic’, ‘nearest’, ‘none’, or any constant value
            like 0 or np.nan. If a list then each element must be a str, tuple or dict defining the boundary for the
            corresponding array in args. The default value is ‘none’. For more information, the user is referred to
            <https://docs.dask.org/en/stable/array-overlap.html#boundaries>`_.
        """
        size = np.prod(arg_shape).item()

        super(Stencil, self).__init__((size, size))

        self.arg_shape = arg_shape
        self.ndim = len(arg_shape)
        self._sanitize_inputs(stencil_coefs, center, boundary)
        self._make_stencils(self.stencil_coefs)

    def _apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._trim_apply(self.stencil(self._pad(arr))).reshape(arr.shape)

    def _apply_dask(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.reshape(-1, *self.arg_shape).map_overlap(
            self.stencil_dask, depth=self._depth, boundary=self._boundary, dtype=pycrt.getPrecision().value
        )

    def _apply_cupy(self, arr: pyct.NDArray) -> pyct.NDArray:
        out_shape = arr.shape
        arr, out = self._pad_and_allocate_output(arr)
        blockspergrid = [math.ceil(out.shape[i] / tpb) for i, tpb in enumerate(self.threadsperblock)]
        self.stencil[blockspergrid, self.threadsperblock](arr, out)
        return self._trim_apply(out).reshape(out_shape)

    def _adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        return self._trim_adjoint(self.stencil_adjoint(self._pad(arr, direction="adjoint"))).reshape(arr.shape)

    def _adjoint_dask(self, arr: pyct.NDArray) -> pyct.NDArray:
        return arr.reshape(-1, *self.arg_shape).map_overlap(
            self.stencil_adjoint_dask,
            depth=self._depth_adjoint,
            boundary=self._boundary,
            dtype=pycrt.getPrecision().value,
        )

    def _adjoint_cupy(self, arr: pyct.NDArray) -> pyct.NDArray:
        out_shape = arr.shape
        arr, out = self._pad_and_allocate_output(arr, direction="adjoint")
        blockspergrid = tuple([math.ceil(arr.shape[i] / tpb) for i, tpb in enumerate(self.threadsperblock)])
        self.stencil_adjoint[blockspergrid, self.threadsperblock](arr, out)
        return self._trim_adjoint(out).reshape(out_shape)

    @pycrt.enforce_precision(i="arr")
    @pycu.redirect("arr", DASK=_apply_dask, CUPY=_apply_cupy)
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Array to be correlated with the kernel.

        Returns
        -------
        out: NDArray
            NDArray with same shape as the input NDArray, correlated with kernel.
        """
        return self._apply(arr)

    @pycrt.enforce_precision(i="arr")
    @pycu.redirect("arr", DASK=_adjoint_dask, CUPY=_adjoint_cupy)
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        r"""
        Parameters
        ----------
        arr: NDArray
            Array to be convolved with the kernel.

        Returns
        -------
        out: NDArray
            NDArray with same shape as the input NDArray, convolved with kernel.
        """
        return self._adjoint(arr)

    def _make_stencils_cpu(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        self.stencil = pycstencil.make_nd_stencil(self.stencil_coefs, self.center)
        self.stencil_dask = pycstencil.make_nd_stencil(self.stencil_coefs_dask, self.center_dask)
        self.stencil_adjoint = pycstencil.make_nd_stencil(self.stencil_coefs_adjoint, self.center_adjoint)
        self.stencil_adjoint_dask = pycstencil.make_nd_stencil(
            self.stencil_coefs_adjoint_dask, self.center_adjoint_dask
        )

    def _make_stencils_gpu(self, stencil_coefs: pyct.NDArray, **kwargs) -> None:
        self.stencil = pycstencil.make_nd_stencil_gpu(self.stencil_coefs, self.center)
        self.stencil_dask = pycstencil.make_nd_stencil(self.stencil_coefs_dask, self.center_dask)
        self.stencil_adjoint = pycstencil.make_nd_stencil_gpu(self.stencil_coefs_adjoint, self.center_adjoint)
        self.stencil_adjoint_dask = pycstencil.make_nd_stencil(
            self.stencil_coefs_adjoint_dask, self.center_adjoint_dask
        )

    @pycu.redirect("stencil_coefs", CUPY=_make_stencils_gpu)
    def _make_stencils(self, stencil_coefs: pyct.NDArray) -> None:
        self._make_stencils_cpu(stencil_coefs)

    def _pad(self, arr: pyct.NDArray, direction: str = "apply") -> pyct.NDArray:
        r"""
        Pad input according to the kernel's shape and center.
        """
        xp = pycu.get_array_module(arr)
        arr = arr.reshape(-1, *self.arg_shape)
        for i in range(1, len(self.pad_widths[direction])):
            padding_kwargs = {key: value[i] for key, value in self._padding_kwargs.items()}
            _pad_width = tuple(
                [(0, 0) if i != j else self.pad_widths[direction][i] for j in range(len(self.pad_widths[direction]))]
            )
            arr = xp.pad(array=arr, pad_width=_pad_width, **padding_kwargs)
        return arr

    def _sanitize_inputs(self, stencil_coefs: pyct.NDArray, center: pyct.NDArray, boundary):
        r"""
        Check that inputs have the correct shape and correctly handle the boundary conditions.
        """
        assert len(center) == stencil_coefs.ndim == self.ndim, (
            "The stencil coefficients should have the same"
            " number of dimensions as `arg_shape` and the "
            "same length as `center`."
        )
        self.xp = xp = pycu.get_array_module(stencil_coefs)
        self.stencil_coefs = self.stencil_coefs_dask = stencil_coefs
        self.center = self.center_dask = xp.asarray(center)
        self.stencil_coefs_adjoint = self.stencil_coefs_adjoint_dask = xp.flip(stencil_coefs)
        self.center_adjoint = self.center_adjoint_dask = xp.array(stencil_coefs.shape) - 1 - xp.asarray(center)

        ndim = stencil_coefs.ndim

        self._set_boundaries(ndim, boundary)
        self._set_depths(ndim)
        self._set_trimmers()
        self.threadsperblock = self._compute_threadsperblock(stencil_coefs.shape)

    def _set_boundaries(self, ndim, boundary):
        default = "none"
        if boundary is None:
            boundary = default
        if not isinstance(boundary, (tuple, dict)):
            boundary = (boundary,) * (ndim + 1)
        if isinstance(boundary, tuple):
            boundary = dict(zip(range(ndim + 1), boundary))
        if isinstance(boundary, dict):
            boundary = {ax + 1: boundary.get(ax, default) for ax in range(ndim)}
            boundary[0] = boundary[1]

        mode = dict()
        cval = dict()

        for ax in range(ndim + 1):
            this_mode = boundary.get(ax)
            if this_mode == "none":
                mode.update(dict([(ax, "constant")]))
                cval.update(dict([(ax, 0.0)]))

            elif this_mode == "periodic":
                mode.update(dict([(ax, "wrap")]))

            elif this_mode == "reflect":
                mode.update(dict([(ax, "reflect")]))

            elif this_mode == "nearest":
                mode.update(dict([(ax, "edge")]))

            elif isinstance(this_mode, numbers.Number):
                mode.update(dict([(ax, "constant")]))
                cval.update(dict([(ax, this_mode)]))
            else:
                raise ValueError(
                    f"`boundary` should be `reflect`, `periodic`, `nearest`, `none` or a constant value,"
                    f" but got {this_mode} instead."
                )
        self._boundary = boundary
        self._padding_kwargs = dict(mode=mode, constant_values=cval)

    def _set_depths(self, ndim):
        xp = self.xp
        depth_right = xp.array(self.stencil_coefs.shape) - self.center - 1
        _pad_width = tuple([(0, 0)] + [(self.center[i].item(), depth_right[i].item()) for i in range(ndim)])
        depth_right = xp.array(self.stencil_coefs_adjoint.shape) - self.center_adjoint - 1
        _pad_width_adjoint = tuple(
            [(0, 0)] + [(self.center_adjoint[i].item(), depth_right[i].item()) for i in range(ndim)]
        )
        self.pad_widths = dict(apply=_pad_width, adjoint=_pad_width_adjoint)

        # If boundary conditions are not 'none' for some dimension, then Dask's map_overlap needs a symmetric kernel.
        if any(map("none".__ne__, self._boundary.values())):  # some key is not 'none' --> center dask kernel
            self._depth, self.stencil_coefs_dask, self.center_dask = self._convert_sym_ker(
                self.stencil_coefs_dask, self.center_dask
            )
            self._depth_adjoint, self.stencil_coefs_adjoint_dask, self.center_adjoint_dask = self._convert_sym_ker(
                self.stencil_coefs_adjoint_dask, self.center_adjoint_dask
            )
        else:
            depth_right = xp.array(self.stencil_coefs_dask.shape) - self.center_dask - 1
            self._depth = {0: 0}
            self._depth.update({i + 1: (self.center_dask[i], depth_right[i]) for i in range(self.ndim)})

            depth_right = xp.array(self.stencil_coefs_adjoint_dask.shape) - self.center_adjoint_dask - 1
            self._depth_adjoint = {0: 0}
            self._depth_adjoint.update({i + 1: (self.center_adjoint_dask[i], depth_right[i]) for i in range(self.ndim)})

    def _set_trimmers(self):
        arg_shape_apply = [s + np.sum(self.pad_widths["apply"][i + 1]) for i, s in enumerate(self.arg_shape)]
        arg_shape_adjoint = [s + np.sum(self.pad_widths["adjoint"][i + 1]) for i, s in enumerate(self.arg_shape)]
        self._trim_apply = TrimOp(arg_shape_apply, self.pad_widths["apply"])
        self._trim_adjoint = TrimOp(arg_shape_adjoint, self.pad_widths["adjoint"])

    def _convert_sym_ker(
        self, stencil_coefs: pyct.NDArray, center: pyct.NDArray
    ) -> typ.Tuple[typ.Tuple, pyct.NDArray, pyct.NDArray]:
        r"""
        Creates a symmetric kernel stencil to use with Dask's map_overlap() in case of non-default ('none') boundary
        conditions.
        """
        xp = pycu.get_array_module(stencil_coefs)
        dist_center = (
            -(xp.array(stencil_coefs.shape) // 2 - xp.asarray(center)) * 2
            - xp.mod(xp.array(stencil_coefs.shape), 2)
            + 1
        )

        pad_left = abs(xp.clip(dist_center, a_min=-xp.infty, a_max=0)).astype(int)
        pad_right = xp.clip(dist_center, a_min=0, a_max=xp.infty).astype(int)
        pad_width = tuple([(pad_left[i].item(), pad_right[i].item()) for i in range(self.ndim)])
        stencil_coefs = xp.pad(stencil_coefs, pad_width=pad_width)
        center = xp.array(stencil_coefs.shape) // 2

        depth_sides = xp.array(stencil_coefs.shape) - center - 1
        _depth = tuple([0] + [depth_sides[i] for i in range(self.ndim)])
        return _depth, stencil_coefs, center

    @staticmethod
    def _next_power_of_2(x):
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def _compute_threadsperblock(self, kernel_shape):
        tpb = [int(self._next_power_of_2(kernel_shape[d])) for d in range(len(kernel_shape) - 1)]
        tpb = [1] + tpb + [int(1024 / np.prod(tpb))]

        # If nthreads larger than array size, use threads in other dims
        for i in range(len(tpb) - 1, 0, -1):
            while tpb[i] > self.arg_shape[i - 1] + np.sum(self.pad_widths["apply"][i]):
                tpb[i] = int(tpb[i] / 2)
                if i > 1:
                    tpb[i - 1] = int(tpb[i - 1] * 2)
        return tpb

    def _pad_and_allocate_output(
        self, arr: pyct.NDArray, direction: str = "apply"
    ) -> typ.Tuple[pyct.NDArray, pyct.NDArray]:
        xp = pycu.get_array_module(arr)
        arr = self._pad(arr, direction=direction)
        out = xp.zeros_like(arr)
        return arr, out


class SumOp(pyca.LinOp):
    """
    TODO Docstring and test
    """

    def __init__(self, arg_shape, axis):
        self.arg_shape = arg_shape
        self.axis = axis

        super(SumOp, self).__init__((np.prod(arg_shape) / arg_shape[axis], np.prod(arg_shape)))

        self.tile = np.ones(len(self.arg_shape) + 1, dtype=int)
        self.tile[self.axis + 1] = self.arg_shape[self.axis]

    @pycrt.enforce_precision(i="arr")
    def apply(self, arr):
        xp = pycu.get_array_module(arr)
        return (
            arr.reshape(-1, *self.arg_shape)
            .sum(axis=self.axis + 1)
            .reshape(-1, xp.prod(self.arg_shape[: self.axis] + self.arg_shape[self.axis + 1 :]))
        )

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr):
        xp = pycu.get_array_module(arr)
        out = xp.expand_dims(
            arr.reshape(-1, *(self.arg_shape[: self.axis] + self.arg_shape[self.axis + 1 :])), self.axis + 1
        )
        out = xp.tile(out, self.tile).reshape(-1, np.prod(self.arg_shape))
        return out
