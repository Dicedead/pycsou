from dataclasses import dataclass

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage

import pyxu.abc as pxa
import pyxu.experimental.xray._xray as xray
import pyxu.info.ptype as pxt
import pyxu.operator.linop.base as pxlb
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.abc import DiffMap, ProxFunc
from pyxu.operator import PositiveOrthant, SquaredL2Norm
from pyxu.opt.solver import PGD
from pyxu.opt.stop import RelError


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

    def run(self, stop_crit=RelError(eps=1e-3), post_process_optres=None):
        data_fidelity = SquaredL2Norm(np.prod(self.ground_truth.shape)).argshift(-self.ground_truth.flatten()) * self.op
        pgd = PGD(data_fidelity, self.regularizer)
        pgd.fit(x0=self.initialisation, stop_crit=stop_crit, track_objective=True, tau=1 / self.diff_lip)
        alpha, _ = pgd.stats()
        alpha = alpha["x"]

        if post_process_optres is not None:
            alpha = post_process_optres(alpha)

        return self.op(alpha).reshape(*self.ground_truth.shape)


side = 512
gamma = 0.3
beta = 15
num_n = 300
num_t = 350
lambda_ = 4
transmittance_ratio = 0.6  # I_center = trans_ratio * I_0


def disk(side=side, radius=side / 4):
    x, y = np.meshgrid(np.linspace(-side / 2, side / 2, side), np.linspace(-side / 2, side / 2, side))
    ground_truth = 1 * (x**2 + y**2 <= (radius**2))
    return ground_truth


def ellipsis(side_a, num_a, side_b, num_b):
    side_a = side_a / 2
    side_b = side_b / 2
    x, y = np.meshgrid(np.linspace(-side_a, side_a, num_a), np.linspace(-side_b, side_b, num_b))
    ground_truth = 1 * ((x / side_a) ** 2 + (y / side_b) ** 2 <= 1)
    return ground_truth


def absorption_coeff(sides, ratio=transmittance_ratio):
    sides = pitch * sides
    return -np.log(ratio) * np.sum(sides) / (2 * np.prod(sides))


def inverse_absorption_map(sides, ratio=transmittance_ratio):
    sides = pitch * sides
    alpha = absorption_coeff(sides, ratio)
    distance_max = sides[0] / 2
    points = np.indices(sides.astype(int)).transpose((1, 2, 0)) * pitch
    dist = np.linalg.norm(points - distance_max, axis=-1)
    distances = distance_max - dist
    correction_map = np.exp(alpha * distances)
    return correction_map


def epfl_logo(path="../res/epfl_square.png", downsample_x=4, downsample_y=4, inverted=False):
    im = iio.imread(path)[::downsample_y, ::downsample_x, 0]
    if inverted:
        im = im.max() - im
    return im / im.max()


def chirp_signal(sides, a, T=1):
    # sin = np.sin(2 * np.pi * np.linspace(0, v_max, num=sides[0]) * np.linspace(0, T, num=sides[0]))
    sin = np.sin(2 * np.pi * a * (np.linspace(0, T, num=sides[0]) ** 2))
    sin = np.sign(sin).clip(0, 1)
    sin = np.tile(sin.reshape(1, -1), (sides[1], 1))
    return sin


def concentric_circles(path="../res/concentric.png", downsample_x=2, downsample_y=2, inverted=False):
    im = iio.imread(path)[::downsample_y, ::downsample_x, 0]
    if inverted:
        im = im.max() - im
    return im / im.max()


ground_truth = epfl_logo(inverted=True)  # chirp_signal([side, side], a=20)  # epfl_logo(inverted=True)

side = np.array(ground_truth.shape)
origin = 0
pitch = np.array([1.0, 1.0])
w_spec = absorption_coeff(side) * ellipsis(pitch[1] * side[1], side[1], pitch[0] * side[0], side[0])

angle = np.linspace(0, 2 * np.pi, num_n, endpoint=False)
t_max = pitch * side / 2  # 10% over ball radius
t_offset = np.linspace(-t_max[0], t_max[1], num_t, endpoint=True)

n = np.stack([np.cos(angle), np.sin(angle)], axis=1)
t = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0

n_spec = np.broadcast_to(n.reshape(num_n, 1, 2), (num_n, num_t, 2))  # (N_angle, N_offset, 2)
t_spec = t.reshape(num_n, 1, 2) * t_offset.reshape(num_t, 1)
t_spec += pitch * side / 2

weighted_xrt = xray.XRayTransform.init(
    arg_shape=ground_truth.shape,
    origin=origin,
    pitch=pitch,
    method="ray-trace",
    n_spec=n_spec.reshape(-1, 2),
    t_spec=t_spec.reshape(-1, 2),
    w_spec=w_spec,
)

unweighted_xrt = xray.XRayTransform.init(
    arg_shape=ground_truth.shape,
    origin=origin,
    pitch=pitch,
    method="ray-trace",
    n_spec=n_spec.reshape(-1, 2),
    t_spec=t_spec.reshape(-1, 2),
    w_spec=None,
)

# hstack([weighted_xrt.T, pxlb.NullFunc(1)])
# beta = 3
# kelly = ReconstructionTechnique(
#     data_fidelity=SquaredL2Norm(np.prod(ground_truth.shape))
#                    .argshift(-ground_truth.flatten()) * SigLU(weighted_xrt.dim)
#     .argscale(beta).argshift(hstack()) * weighted_xrt.T,
# )

lcav = ReconstructionTechnique(
    ground_truth=ground_truth,
    op=weighted_xrt.T,
    regularizer=PositiveOrthant(weighted_xrt.codim),
    initialisation=0 * np.random.randn(weighted_xrt.codim),
    diff_lip=400000,
)

moser1 = ReconstructionTechnique(
    ground_truth=ground_truth,
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(unweighted_xrt.codim),
    initialisation=0 * np.random.randn(unweighted_xrt.codim),
    diff_lip=400000,
)

moser2 = ReconstructionTechnique(
    ground_truth=inverse_absorption_map(side) * ground_truth,
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(unweighted_xrt.codim),
    initialisation=0 * np.random.randn(unweighted_xrt.codim),
    diff_lip=400000,
)


def plot_four_images(imgs, subtitles, main_title, file_title, processing=None):
    if processing is None:
        processing = [lambda x: x] * 4

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax_flat = ax.flatten()
    fig.suptitle(main_title, fontsize=16)
    fig.subplots_adjust(hspace=0.5)

    for idx, a in enumerate(ax_flat):
        im = a.imshow(processing[idx](imgs[idx]), cmap="Greys", vmin=0, vmax=1)
        a.set_title(subtitles[idx])
        plt.colorbar(im, ax=a)

    fig.savefig(file_title, dpi=500)
    plt.close(fig)


def threshold_processing(image):
    otsu = skimage.filters.threshold_otsu(image, nbins=2)
    print(otsu)
    res = image.copy()
    res[image < otsu] = 0
    res[image >= otsu] = 1
    return res


if __name__ == "__main__":
    moser1_img = moser1.run()
    moser2_img = moser2.run()
    lcav_img = lcav.run()

    plot_four_images(
        [ground_truth, moser1_img, moser2_img, lcav_img],
        ["Ground truth", "Moser1", "Moser2", "LCAV"],
        "No thresholding",
        "no_thresholding.png",
    )

    plot_four_images(
        [ground_truth, moser1_img, moser2_img, lcav_img],
        ["Ground truth", "Moser1", "Moser2", "LCAV"],
        "With thresholding",
        "with_thresholding.png",
        processing=[threshold_processing] * 4,
    )
