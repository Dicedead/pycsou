import drjit as dr
import mitsuba as mi

import pycsou.abc as pyca
import pycsou.util.ptype as pyct


class RaytracingOp(pyca.LinOp):

    def __init__(
            self,
            scene: dict,
            params: list[str],
            emitters: list[dict] | None = None,
            sensors: list[dict] | None = None,
            mitsuba_variant: str = "cuda_ad_rgb"  # TODO in doc if none: don't set variant
    ):
        if mitsuba_variant is not None:
            if mitsuba_variant not in mi.variants():
                raise ValueError(f"[RaytracingOp] Selected Mitsuba variant: {mitsuba_variant} not supported.\n"
                                 f"Supported variants: {mi.variants()}")
            mi.set_variant(mitsuba_variant)

        scene = mi.load_dict(scene)

        if emitters is None:
            emitters = scene.emitters()

        if sensors is None:
            sensors = scene.sensors()

        self.sensors = sensors
        self.scene = scene
        self.emitters = emitters
        # TODO params data shape / size

        shape = (-1, len(sensors))  # TODO infer shape from parameters
        super().__init__(shape)

    def adjoint(self, arr: pyct.NDArray) -> pyct.NDArray:
        pass

    def apply(self, arr: pyct.NDArray) -> pyct.NDArray:
        pass
