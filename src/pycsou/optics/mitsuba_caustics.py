import functools

import drjit as dr
import numpy as np

import mitsuba as mi
import pycsou.abc as pyca
import pycsou.util.ptype as pyct


class MitsubaLossWrapper(pyca.DiffFunc):
    def __init__(
        self,
        scene,
        target,
        heightmap_shape,
        loss_func,
        spp=128,
        initial_heightmap=None,
    ):
        heightmap_size = functools.reduce(lambda x, y: x * y, heightmap_shape, 1)
        super().__init__(shape=(1, heightmap_size))
        self.__loss = loss_func
        self.__target = target
        self.__scene = scene
        self.__heightmap_shape = heightmap_shape

        self.__params_scene = mi.traverse(scene)

        if initial_heightmap is None:
            initial_heightmap = dr.zeros(mi.TensorXf, heightmap_shape)

        self.__heightmap_texture = mi.load_dict(
            {
                "type": "bitmap",
                "id": "heightmap_texture",
                "bitmap": mi.Bitmap(initial_heightmap),
                "raw": True,
            }
        )
        self.__params = mi.traverse(self.__heightmap_texture)
        self.__params.keep(["data"])

        self.__positions_initial = dr.unravel(mi.Vector3f, self.__params_scene["lens.vertex_positions"])
        self.__normals_initial = dr.unravel(mi.Vector3f, self.__params_scene["lens.vertex_normals"])
        self.__lens_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(self.__positions_initial))
        self.__lens_si.uv = dr.unravel(type(self.__lens_si.uv), self.__params_scene["lens.vertex_texcoords"])
        self.__spp = spp

        self.__it = 0
        self.__last_image = mi.render(self.__scene, self.__params, seed=0, spp=2 * self.__spp, spp_grad=self.__spp)

    def __apply_displacement(self, amplitude=1.0):
        # Enforce reasonable range. For reference, the receiving plane
        # is 7 scene units away from the lens.
        vmax = 1 / 100.0
        self.__params["data"] = dr.clamp(self.__params["data"], -vmax, vmax)
        dr.enable_grad(self.__params["data"])

        height_values = self.__heightmap_texture.eval_1(self.__lens_si)
        new_positions = height_values * self.__normals_initial * amplitude + self.__positions_initial
        self.__params_scene["lens.vertex_positions"] = dr.ravel(new_positions)
        self.__params_scene.update()

    def grad(self, arr: pyct.NDArray) -> pyct.NDArray:
        loss = self.apply(arr)
        dr.backward(loss)
        return dr.grad(self.__params["data"]).numpy().flatten()

    def apply(self, arr: pyct.NDArray, seed=None) -> pyct.NDArray:
        self.__it += 1
        if seed is None:
            seed = self.__it

        self.__params.update({"data": mi.TensorXf(array=arr, shape=self.__heightmap_shape)})
        self.__apply_displacement()
        self.__last_image = mi.render(self.__scene, self.__params, seed=seed, spp=2 * self.__spp, spp_grad=self.__spp)
        return self.__loss(self.__last_image, self.__target)

    def get_last_image(self):
        return self.__last_image

    def asloss(self, data: pyct.NDArray = None) -> pyct.OpT:
        raise NotImplementedError(
            "[MitsubaLossWrapper] asloss function is not implemented as it"
            "                     depends on the particular given loss function"
        )

    def lipschitz(self, **kwargs) -> pyct.Real:
        return np.infty  # explicit over implicit

    def diff_lipschitz(self, **kwargs) -> pyct.Real:
        return np.infty
