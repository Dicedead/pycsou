import functools
from typing import Optional

import drjit as dr

import mitsuba as mi
import pycsou.abc as pyca
import pycsou.runtime as pycrt
import pycsou.util.ptype as pyct
from mitsuba.python.util import SceneParameters

__all__ = ["RaytracingOp"]


class RaytracingOp(pyca.LinOp):
    def __init__(
        self,
        scene: dict | str,  # TODO doc: str is for XML, dict gets loaded
        params: str | SceneParameters,
        emitters: Optional[list[dict]] = None,
        sensors: Optional[list[dict]] = None,
        # mitsuba_variant: str = "cuda_ad_rgb"
        # TODO doc: if none: don't set variant + see if this arg is relevant or should be inferred from Pycsou
    ):
        # if mitsuba_variant is not None:
        #     if mitsuba_variant not in mi.variants():
        #         raise ValueError(
        #             f"[RaytracingOp] Selected Mitsuba variant: {mitsuba_variant} not supported.\n"
        #             f"Supported variants: {mi.variants()}"
        #         )
        #     mi.set_variant(mitsuba_variant)

        if isinstance(scene, dict):
            scene = mi.load_dict(scene)
        else:
            scene = mi.load_file(scene)

        if emitters is None:
            emitters = scene.emitters()

        if sensors is None:
            sensors = scene.sensors()

        self.sensors = sensors
        self.scene = scene
        self.emitters = emitters
        self.scene_params = mi.traverse(scene)
        self.params = params if isinstance(params, SceneParameters) else self.scene_params[params]
        self.argshape = self.params["data"].shape if isinstance(params, SceneParameters) else (1,)

        product_size = functools.reduce(lambda x, y: x * y, list(self.argshape), 1)
        shape = (product_size, len(sensors))
        super().__init__(shape)

    @pycrt.enforce_precision(i="arr")
    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        pass

    # @pycrt.enforce_precision(i="arr")
    # TODO (note) can't here! Cannot coerce <class 'drjit.cuda.ad.TensorXf'> to scalar/array of precision float64

    # note: no access to what sensors see individually, except: by rendering each individually
    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        self.params["data"] = arr
        dr.enable_grad(self.params["data"])
        self.scene_params.update()
        return self.render()

    def render(self) -> mi.TensorXf:
        return mi.render(self.scene)
