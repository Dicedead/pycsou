import collections.abc as cabc
import datetime as dt
import typing as typ
import warnings

import numpy as np

import pycsou.abc as pyca
import pycsou.util as pycu
import pycsou.util.ptype as pyct

__all__ = [
    "AbsError",
    "ManualStop",
    "MaxDuration",
    "MaxIter",
    "Memorize",
    "RelError",
]


SVFunction = typ.Union[
    cabc.Callable[[pyct.Real], pyct.Real],
    cabc.Callable[[pyct.NDArray], pyct.NDArray],
]


class MaxIter(pyca.StoppingCriterion):
    """
    Stop iterative solver after a fixed number of iterations.

    Tip
    ---
    If you want to add a grace period to a solver, i.e. for it to do *at least* N iterations before
    stopping based on the value of another criteria, you can AND `MaxIter` with the other criteria.

    Example
    -------

    >>> sc = MaxIter(n=5) & AbsError(eps=0.1)
    # If N_iter < 5 -> never stop.
    # If N_iter >= 5 -> stop if AbsError() decides to.
    """

    def __init__(self, n: pyct.Integer):
        """
        Parameters
        ----------
        n: pyct.Integer
            Max number of iterations allowed.
        """
        try:
            assert int(n) > 0
            self._n = int(n)
        except:
            raise ValueError(f"n: expected positive integer, got {n}.")
        self._i = 0

    def stop(self, state: cabc.Mapping) -> bool:
        self._i += 1
        return self._i > self._n

    def info(self) -> cabc.Mapping[str, float]:
        return dict(N_iter=self._i)

    def clear(self):
        self._i = 0


class ManualStop(pyca.StoppingCriterion):
    """
    Continue-forever criterion.

    This class is useful when calling `Solver.fit` with mode=MANUAL/ASYNC to defer the stopping
    decision to an explicit call by the user, i.e.:
    * mode=MANUAL: user must stop calling `next(solver.steps())`;
    * mode=ASYNC: user must call `Solver.stop`.
    """

    def stop(self, state: cabc.Mapping) -> bool:
        return False

    def info(self) -> cabc.Mapping[str, float]:
        return dict()

    def clear(self):
        pass


class MaxDuration(pyca.StoppingCriterion):
    """
    Stop iterative solver after a specified duration has elapsed.
    """

    def __init__(self, t: dt.timedelta):
        """
        Parameters
        ----------
        t: dt.timedelta
            Max runtime allowed.
        """
        try:
            assert t > dt.timedelta()
            self._t_max = t
        except:
            raise ValueError(f"t: expected positive duration, got {t}.")
        self._t_start = dt.datetime.now()
        self._t_now = self._t_start

    def stop(self, state: cabc.Mapping) -> bool:
        self._t_now = dt.datetime.now()
        return (self._t_now - self._t_start) > self._t_max

    def info(self) -> cabc.Mapping[str, float]:
        d = (self._t_now - self._t_start).total_seconds()
        return dict(duration=d)

    def clear(self):
        self._t_start = dt.datetime.now()
        self._t_now = self._t_start


class Memorize(pyca.StoppingCriterion):
    """
    Memorize a variable.
    (Special StoppingCriterion mostly useful for tracking objective functions in Solver.)
    """

    def __init__(self, var: pyct.VarName):
        """
        Parameters
        ----------
        var: pyct.VarName
            Variable in `Solver._mstate` to query.
            Must be a scalar or NDArray (1D).
        """
        self._var = var
        self._val = np.r_[0]  # last memorized value in stop().

    def stop(self, state: cabc.Mapping) -> bool:
        x = state[self._var]
        if isinstance(x, pyct.Real):
            x = np.r_[x]
        assert x.ndim == 1

        self._val = pycu.compute(x)
        return False

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"Memorize[{self._var}]": float(self._val.max())}  # takes the only element available.
        else:
            data = {
                f"Memorize[{self._var}]_min": float(self._val.min()),
                f"Memorize[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]


class AbsError(pyca.StoppingCriterion):
    """
    Stop iterative solver after absolute norm of a variable (or function thereof) reaches threshold.
    """

    def __init__(
        self,
        eps: pyct.Real,
        var: pyct.VarName = "x",
        f: SVFunction = None,
        norm: pyct.Real = 2,
        satisfy_all: bool = True,
    ):
        """
        Parameters
        ----------
        eps: pyct.Real
            Positive threshold.
        var: pyct.VarName
            Variable in `Solver._mstate` to query.
        f: Callable
            Optional function to pre-apply to `Solver._mstate[var]` before applying the norm.
            Defaults to the identity function. The callable should either:
            * accept a scalar input -> output a scalar, or
            * accept an NDArray input -> output an NDArray, i.e same semantics as `Operator.apply`.
        norm: pyct.Integer | pyct.Real
            Ln norm to use >= 0. (Default: L2.)
        satisfy_all: bool
            If True (default) and `Solver._mstate[var]` is multi-dimensional, stop if all evaluation
            points lie below threshold.
        """
        try:
            assert eps > 0
            self._eps = eps
        except:
            raise ValueError(f"eps: expected positive threshold, got {eps}.")

        self._var = var
        self._f = f if (f is not None) else (lambda _: _)

        try:
            assert norm >= 0
            self._norm = norm
        except:
            raise ValueError(f"norm: expected non-negative, got {norm}.")

        self._satisfy_all = satisfy_all
        self._val = np.r_[0]  # last computed Ln norm(s) in stop().

    def stop(self, state: cabc.Mapping) -> bool:
        fx = self._f(state[self._var])
        if isinstance(fx, pyct.Real):
            fx = np.r_[fx]
        xp = pycu.get_array_module(fx)

        self._val = xp.linalg.norm(fx, ord=self._norm, axis=-1, keepdims=True)
        rule = xp.all if self._satisfy_all else xp.any
        decision = rule(self._val <= self._eps)

        self._val, decision = pycu.compute(self._val, decision)
        return decision

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"AbsError[{self._var}]": float(self._val.max())}  # takes the only element available.
        else:
            data = {
                f"AbsError[{self._var}]_min": float(self._val.min()),
                f"AbsError[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]


class RelError(pyca.StoppingCriterion):
    """
    Stop iterative solver after relative norm change of a variable (or function thereof) reaches
    threshold.
    """

    def __init__(
        self,
        eps: pyct.Real,
        var: pyct.VarName = "x",
        f: SVFunction = None,
        norm: pyct.Real = 2,
        satisfy_all: bool = True,
    ):
        """
        Parameters
        ----------
        eps: pyct.Real
            Positive threshold.
        var: pyct.VarName
            Variable in `Solver._mstate` to query.
        f: Callable
            Optional function to pre-apply to `Solver._mstate[var]` before applying the norm.
            Defaults to the identity function. The callable should either:
            * accept a scalar input -> output a scalar, or
            * accept an NDArray input -> output an NDArray, i.e same semantics as `Operator.apply`.
        norm: pyct.Integer | pyct.Real
            Ln norm to use >= 0. (Default: L2.)
        satisfy_all: bool
            If True (default) and `Solver._mstate[var]` is multi-dimensional, stop if all evaluation
            points lie below threshold.
        """
        try:
            assert eps > 0
            self._eps = eps
        except:
            raise ValueError(f"eps: expected positive threshold, got {eps}.")

        self._var = var
        self._f = f if (f is not None) else (lambda _: _)

        try:
            assert norm >= 0
            self._norm = norm
        except:
            raise ValueError(f"norm: expected non-negative, got {norm}.")

        self._satisfy_all = satisfy_all
        self._val = np.r_[0]  # last computed Ln rel-norm(s) in stop().
        self._x_prev = None  # buffered var from last query.

    def stop(self, state: cabc.Mapping) -> bool:
        x = state[self._var]
        if isinstance(x, pyct.Real):
            x = np.r_[x]
        xp = pycu.get_array_module(x)

        if self._x_prev is None:
            self._x_prev = x.copy()
            # force 1st .info() call to have same format as further calls.
            self._val = np.zeros(shape=(1,) if (x.ndim == 1) else x.shape[:-1])
            return False  # decision deferred: insufficient history to evaluate rel-err.
        else:
            norm = lambda _: xp.linalg.norm(_, ord=self._norm, axis=-1, keepdims=True)
            rule = xp.all if self._satisfy_all else xp.any

            fx_prev = self._f(self._x_prev)
            numerator = norm(self._f(x) - fx_prev)
            denominator = norm(fx_prev)
            decision = rule(numerator <= self._eps * denominator)

            with warnings.catch_warnings():
                # Store relative improvement values for info(). Special care must be taken for the
                # problematic case 0/0 -> NaN.
                warnings.simplefilter("ignore")
                self._val = numerator / denominator
                self._val[xp.isnan(self._val)] = 0  # no relative improvement.
            self._x_prev = x.copy()

            self._x_prev, self._val, decision = pycu.compute(self._x_prev, self._val, decision)
            return decision

    def info(self) -> cabc.Mapping[str, float]:
        if self._val.size == 1:
            data = {f"RelError[{self._var}]": float(self._val.max())}  # takes the only element available.
        else:
            data = {
                f"RelError[{self._var}]_min": float(self._val.min()),
                f"RelError[{self._var}]_max": float(self._val.max()),
            }
        return data

    def clear(self):
        self._val = np.r_[0]
        self._x_prev = None


class StopCriterion_LSQMR(pycs.StoppingCriterion):
    r"""
    Stop iterative solver of :py:class:`pycsou.opt.solver.lsqr.LSQR` or :py:class:`pycsou.opt.solver.lsmr.LSMR` after
    certain customized conditions reach.

    **Stopping Tests:**

    **1.** Iteration number reach `max_iter` before other stopping conditions is satisfied.

    **2.** When `x` is an approximate solution to `A@x = B`, according to `atol` and `btol`.

    **3.** When `x` approximately solves the least-squares problem according to `atol`.

    **4.** When :math:`\text{cond}(A)` is greater than :math:`\text{conlim}`.

    **5.** Same as **2** with :math:`\text{atol} = \text{btol} = \text{eps}` (machine precision)

    **6.** Same as **3** with :math:`\text{atol} = \text{eps}`.

    **7.** Same as **4** with :math:`\text{conlim} = 1/\text{eps}`.
    """

    def __init__(
        self,
        method: str,
        atol: float,
        ctol: float,
        itn: int,
        iter_lim: int,
    ):
        """
        Parameters
        ----------
        method: str
            Solver method. Either "lsqr" or "lsmr", otherwise raises error.
        atol, ctol: float
            Stopping tolerances.
        itn: int
            Iteration number.
        iter_lim: int
            Iteration limit.
        """
        self._method = method
        self._atol, self._ctol = atol, ctol
        self._itn, self._iter_lim = itn, iter_lim
        self._istop = None
        self._x0 = self._test1 = self._test2 = None
        self._normA = self._condA = None
        if self._method == "lsqr":
            self._normr1 = self._normr2 = None
        elif self._method == "lsmr":
            self._normr = self._normar = None
        else:
            raise ValueError(f"method: expected 'lsqr' or 'lsmr', got {method}.")

    def stop(self, state: cabc.Mapping) -> bool:

        # Check if there's a trivial solution
        if state["trivial"].all():
            return True

        # Update parameters here to update info:
        self._x0, self._test1, self._test2 = state["x"][0], state["test1"], state["test2"]
        self._normA = state["normA"]
        self._condA = state["condA"]

        # Parameters specific to lsqr/lsmr
        if self._method == "lsqr":
            self._normr1, self._normr2 = state["normr1"], state["normr2"]
        else:
            self._normr, self._normar = state["normr"], state["normar"]

        # If iteration number is 0:
        if self._itn == 0:
            self._itn += 1
            return False
        else:
            try:
                self._x0 = self._x0[0]
            except:
                pass

        # Parameters to test stopping criterions
        test1, test2, test3 = state["test1"], state["test2"], state["test3"]
        t1, rtol = state["t1"], state["rtol"]

        # Update self._istop
        self._istop = -np.ones_like(test1)

        # Applying tests:
        if self._itn >= self._iter_lim:
            self._istop[:] = 7
        self._istop[1 + test3 <= 1] = 6
        self._istop[test3 <= self._ctol] = 3
        self._istop[1 + test2 <= 1] = 5
        self._istop[test2 <= self._atol] = 2
        self._istop[test1 <= rtol] = 1
        self._istop[1 + t1 <= 1] = 4

        self._itn += 1

        # Getting and returning decision:
        decision = True if (self._istop != -1).all() else False
        if decision:
            # If there's any trivial solution, add the trivial solution to the solution
            if state["trivial"].any():
                xp = pycu.get_array_module(state["x"])
                temp_x = state["x"]
                state["x"] = xp.zeros((*state["trivial"].squeeze().shape, state["x"].shape[-1]))
                state["x"][xp.invert(state["trivial"].squeeze())] = temp_x
        return decision

    def info(self) -> cabc.Mapping[str, float]:
        r"""
        **Information given at each iterations:**

        * **x[0] (for first data):** First element of first input data.

        * **norm r1 or norm r:** `norm(b-Ax)` for both methods, "lsqr" and "lsmr".

        * **norm r2 or norm Ar:** `sqrt(norm(r)^2 + damp^2 * norm(x-x0)^2)` or `norm(A^H (b - Ax))`, respectively. The former is given if the method is "lsqr", while the latter is given if the method is "lsmr".

        * **Compatible:** Test1 score, which is calculated to measure how close `x` is an approximated as a solution to `A@x = B`, according to `atol` and `btol`.

        * **LS:** Test2 score, which is calculated to measure how close `x` is approximated as a solution to the least-squares problem according to `atol`.

        * **Norm A:** Estimate of `norm(A)`.

        * **Cond A:** Estimate of `cond(A)`.

        Notes
        -----

        * In case of multiple inputs in parallel, information of only the first input is given for the clear visualization purpose.

        """

        if self._method == "lsqr":
            data = {
                f"x[0]": float(self._x0.ravel()[0]),
                f"norm r1": float(self._normr1.max()),
                f"norm r2": float(self._normr2.max()),
                f"Compatible": float(self._test1.max()),
                f"LS": float(self._test2.max()),
                f"Norm A": float(self._normA.max()),
                f"Cond A": float(self._condA.max()),
            }
        elif self._method == "lsmr":
            data = {
                f"x[0]": float(self._x0.ravel()[0]),
                f"norm r": float(self._normr.max()),
                f"norm Ar": float(self._normar.max()),
                f"Compatible": float(self._test1.max()),
                f"LS": float(self._test2.max()),
                f"Norm A": float(self._normA.max()),
                f"Cond A": float(self._condA.max()),
            }

        return data
