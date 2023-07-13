from abc import abstractmethod

import pycsou.abc as pyca
import pycsou.util.ptype as pyct


class FinSupFunc(pyca.Func):
    @abstractmethod
    def supportF(self, **kwargs) -> pyct.Real:
        return NotImplementedError

    @abstractmethod
    def support(self, **kwargs) -> pyct.Real:
        return NotImplementedError

    @abstractmethod
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplementedError

    @abstractmethod
    def applyF(self, arr: pyct.NDArray) -> pyct.NDArray:
        return NotImplementedError
