from abc import abstractmethod

import pyxu.abc as pyxa
import pyxu.info.ptype as pyxt


class FiniteSupportPulse(pyxa.Func):
    @abstractmethod
    def supportF(self, **kwargs) -> pyxt.Real:
        return NotImplementedError

    @abstractmethod
    def support(self, **kwargs) -> pyxt.Real:
        return NotImplementedError

    @abstractmethod
    def apply(self, arr: pyxt.NDArray) -> pyxt.NDArray:
        return NotImplementedError

    @abstractmethod
    def applyF(self, arr: pyxt.NDArray) -> pyxt.NDArray:
        return NotImplementedError
