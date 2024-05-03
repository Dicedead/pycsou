from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import skimage

import pyxu.abc as pxa
import pyxu.experimental.xray as xray
import pyxu.info.ptype as pxt
import pyxu.operator.linop.base as pxlb
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.abc import DiffMap, ProxFunc
from pyxu.operator import PositiveOrthant, SquaredL2Norm
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError


class SigLU(pxa.DiffMap):
    def __init__(self, dim: pxt.Integer):
        super().__init__(shape=(dim, dim))
        self.lipschitz = 1 / 4
        self.diff_lipschitz = 1 / (6 * np.sqrt(3))

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        x = -arr
        xp.exp(x, out=x)
        x += 1
        x = 1 / x
        x[arr < 0] = 0.25 * arr[arr < 0] + 0.5
        return x

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        x = self.apply(arr)
        v = x.copy()
        x -= 1
        v *= x
        v[arr < 0] = 0.25
        return pxlb.DiagonalOp(v)


@dataclass
class ReconstructionTechnique:
    ground_truth: pxt.NDArray
    op: DiffMap
    regularizer: ProxFunc
    initialisation: pxt.NDArray
    diff_lip: float

    def run(self, save_file, stop_crit=RelError(eps=1e-3) | MaxIter(60), post_process_optres=None):
        data_fidelity = SquaredL2Norm(np.prod(self.ground_truth.shape)).argshift(-self.ground_truth.flatten()) * self.op
        pgd = PGD(data_fidelity, self.regularizer)
        pgd.fit(x0=self.initialisation, stop_crit=stop_crit, track_objective=True, tau=1 / self.diff_lip)
        alpha, _ = pgd.stats()
        alpha = alpha["x"]
        np.save(save_file, alpha)

        if post_process_optres is not None:
            alpha = post_process_optres(alpha)

        return self.op(alpha).reshape(*self.ground_truth.shape)


def sphere(radius, xy_side, z_side, xy_pixels, z_pixels, center):
    x, y, z = np.meshgrid(
        np.linspace(-xy_side / 2, xy_side / 2, xy_pixels, endpoint=True),
        np.linspace(-xy_side / 2, xy_side / 2, xy_pixels, endpoint=True),
        np.linspace(0, z_side, z_pixels, endpoint=False),
    )
    ground_truth = 1 * ((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= (radius**2))
    return ground_truth


def hollow_sphere(inner_radius, outer_radius, xy_side, z_side, xy_pixels, z_pixels, center):
    return sphere(outer_radius, xy_side, z_side, xy_pixels, z_pixels, center) - sphere(
        inner_radius, xy_side, z_side, xy_pixels, z_pixels, center
    )


def nut(path="npys/nut_zres100.npy"):
    return 1 * np.load(path)


def bunny(path="npys/bunny_zres_100.npy"):
    return 1 * np.load(path)


def bunny_reweighted(path="npys/bunny_zres_100_reweighted.npy"):
    return 1 * np.load(path)


def benchy(path="npys/benchy_zres_200.npy"):
    return 1 * np.load(path)


xy_pixels = 100
z_pixels = 100

cylinder_inner_radius = 15.5e-3
cylinder_outer_radius = 16.5e-3
cylinder_max_height = 0.0105216 * 3
cylinder_min_height = 0
assert cylinder_inner_radius < cylinder_outer_radius

sphere_inner = cylinder_inner_radius / 3
sphere_outer = cylinder_inner_radius / 2
center = np.r_[0, 0, cylinder_max_height / 2]

num_n = 1000
bin_size = 1
slm_pixels_height = 100
slm_pixels_width = 200

origin = 0
vox_side = 13.7e-5
max_height = cylinder_max_height
max_offset = cylinder_outer_radius / 10
pitch = vox_side * np.array([1.0, 1.0, 1.0])

# ground_truth = hollow_sphere(sphere_inner, sphere_outer, cylinder_outer_radius, cylinder_max_height, xy_pixels, z_pixels, center)
print("Loading ground truth...")
ground_truth = bunny_reweighted()

print("Creating rays...")
num_heights = slm_pixels_height // bin_size
num_offsets = slm_pixels_width // bin_size

angle = np.linspace(0, 2 * np.pi, num_n, endpoint=False)
heights = np.linspace(0.0000001 + cylinder_min_height, max_height, num_heights, endpoint=False)
t_offset = np.linspace(-max_offset, max_offset, num_offsets, endpoint=True)

n = np.stack([np.cos(angle), np.sin(angle), np.zeros(num_n)], axis=1)
t = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0
t = np.tile(t, len(heights)).reshape(-1, 2)
heights = np.tile(heights, num_n).reshape(-1, 1)
t = np.hstack([t, heights]).reshape(num_n * num_heights, 1, 3)

n_spec = np.broadcast_to(n.reshape(num_n, 1, 3), (num_n, num_offsets, 3))  # (N_angle, N_offset, 3)
n_spec = np.tile(n_spec, num_heights).reshape((num_n * num_heights, num_offsets, 3))
t_spec = t * t_offset.reshape(num_offsets, 1)
t_spec[:, :, -1] = t[:, :, -1]
t_spec += np.r_[pitch[:2], 0] * xy_pixels / 2

n_spec = n_spec.reshape(-1, 3)
t_spec = t_spec.reshape(-1, 3)

print("Building operator...")
unweighted_xrt = xray.XRayTransform.init(
    arg_shape=ground_truth.shape,
    origin=origin,
    pitch=pitch,
    method="ray-trace",
    n_spec=n_spec,
    t_spec=t_spec,
    w_spec=None,
)

print("Diagnostic plot...")
# fig = unweighted_xrt.diagnostic_plot()
# fig.savefig("./diag.png")

lcav = ReconstructionTechnique(
    ground_truth=ground_truth,
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(unweighted_xrt.codim),
    initialisation=0 * np.random.randn(unweighted_xrt.codim),
    diff_lip=5,
)


def threshold_processing(image, divider=1.0):
    otsu = skimage.filters.threshold_otsu(image, nbins=2) / divider
    print(otsu)
    res = image.copy()
    res[image < otsu] = 0
    res[image >= otsu] = 1
    return res


def run():
    save_file = "../solutions/alpha.npy"
    lcav_img = lcav.run(save_file)
    lcav_img = threshold_processing(lcav_img)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.voxels(lcav_img)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.voxels(ground_truth)

    fig.savefig("3dtests.png")


if __name__ == "__main__":
    run()
    lcav_img = np.load("../solutions/alpha.npy")
    lcav_img = unweighted_xrt.adjoint(lcav_img).reshape(ground_truth.shape)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ax1.voxels(threshold_processing(lcav_img))

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ax2.voxels(ground_truth)

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    ax3.voxels(threshold_processing(lcav_img, 4))

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    ax4.voxels(threshold_processing(lcav_img, 40))

    fig.savefig("3dtests.png")
