from dataclasses import dataclass

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage

import pyxu.abc as pxa
import pyxu.experimental.xray._xray as xray
import pyxu.info.ptype as pxt
from pyxu.abc import DiffMap, ProxFunc
from pyxu.experimental.xray._rt import RayXRT
from pyxu.operator import PositiveOrthant, SquaredL2Norm
from pyxu.opt.solver import PGD
from pyxu.opt.stop import RelError


class Loss(pxa.DiffFunc):
    def __init__(self, mask: pxt.NDArray, dh: pxt.Real, dl: pxt.Real, mu: pxt.Real, xrt: RayXRT):
        super().__init__(xrt.shape)
        self._mask = mask
        self._xrt = xrt
        self._loss_fg = PositiveOrthant(dim_shape=mask.sum()).argshift(-dh).moreau_envelope(mu)
        self._loss_bg = PositiveOrthant(dim_shape=(~mask).sum()).argshift(-dl).argscale(-1).moreau_envelope(mu)

    def apply(self, arr):
        L = self._loss_fg.apply(arr[self._mask]) + self._loss_bg.apply(arr[~self._mask])
        return L

    def grad(self, arr):
        y = np.zeros_like(arr)
        y[self._mask] = self._loss_fg.grad(arr[self._mask])
        y[~self._mask] = self._loss_bg.grad(arr[~self._mask])
        return y


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
num_n = 1000
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


def bunny_high():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__088.png")


def bunny_middle1():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__056.png")


def bunny_middle2():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__033.png")


def bunny_low():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__009.png")


ground_truth = bunny_high()  # chirp_signal([side, side], a=20)  # epfl_logo(inverted=True)

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


lcav_low = ReconstructionTechnique(
    ground_truth=bunny_low(),
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(weighted_xrt.codim),
    initialisation=0 * np.random.randn(weighted_xrt.codim),
    diff_lip=400000,
)

lcav_high = ReconstructionTechnique(
    ground_truth=bunny_high(),
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(weighted_xrt.codim),
    initialisation=0 * np.random.randn(weighted_xrt.codim),
    diff_lip=400000,
)

lcav_middle1 = ReconstructionTechnique(
    ground_truth=bunny_middle1(),
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(weighted_xrt.codim),
    initialisation=0 * np.random.randn(weighted_xrt.codim),
    diff_lip=400000,
)

lcav_middle2 = ReconstructionTechnique(
    ground_truth=bunny_middle2(),
    op=unweighted_xrt.T,
    regularizer=PositiveOrthant(weighted_xrt.codim),
    initialisation=0 * np.random.randn(weighted_xrt.codim),
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
    lcav_low_img = lcav_low.run()
    lcav_middle1_img = lcav_middle1.run()
    lcav_middle2_img = lcav_middle2.run()
    lcav_high_img = lcav_high.run()

    plot_four_images(
        [bunny_low(), lcav_low_img, bunny_high(), lcav_high_img],
        ["GT low", "Low", "GT high", "High"],
        "No thresholding",
        "no_thresholding_part1.png",
    )

    plot_four_images(
        [bunny_low(), lcav_low_img, bunny_high(), lcav_high_img],
        ["GT low", "Low", "GT high", "High"],
        "With thresholding",
        "with_thresholding_part1.png",
        processing=[lambda x: x, threshold_processing, lambda x: x, threshold_processing],
    )

    plot_four_images(
        [bunny_middle1(), lcav_middle1_img, bunny_middle2(), lcav_middle2_img],
        ["GT mid 1", "Mid 1", "GT mid 2", "Mid 2"],
        "No thresholding",
        "no_thresholding_part2.png",
    )

    plot_four_images(
        [bunny_middle1(), lcav_middle1_img, bunny_middle2(), lcav_middle2_img],
        ["GT mid 1", "Mid 1", "GT mid 2", "Mid 2"],
        "With thresholding",
        "with_thresholding_part2.png",
        processing=[lambda x: x, threshold_processing, lambda x: x, threshold_processing],
    )
