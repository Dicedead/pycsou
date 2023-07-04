import pycsou.abc as pyca
import pycsou.util.ptype as pyct


class FinSupFunc(pyca.Func):
    def supportF(self, **kwargs) -> pyct.Real:
        return NotImplementedError

    def support(self, **kwargs) -> pyct.Real:
        return NotImplementedError

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplementedError

    def applyF(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplementedError
