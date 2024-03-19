import warnings
from dataclasses import dataclass

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import skimage

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.linop.xrt.ray as xray
from pyxu.abc import ProxFunc
from pyxu.operator import DiagonalOp, PositiveOrthant, SquaredL2Norm
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError

warnings.filterwarnings("ignore")

num_n = 1000
num_t = 350
lambda_ = 0.004
b_param = 10
diff_lip = 400000 * b_param / 2


class BhattInspiredLoss(pxa.DiffFunc):
    def __init__(self, xrt: xray.RayXRT, gt: pxt.NDArray, b: pxt.Real):
        super().__init__(dim_shape=xrt.codim_shape, codim_shape=(1,))
        fg_constant = DiagonalOp(gt)
        bg_constant = DiagonalOp(1 - gt)

        self._loss_fg = SquaredL2Norm(dim_shape=xrt.dim_shape).argshift(-gt) * (fg_constant * xrt.T)
        self._loss_bg = b * SquaredL2Norm(dim_shape=xrt.dim_shape).argshift(-gt) * (bg_constant * xrt.T)

    def apply(self, arr):
        return self._loss_fg.apply(arr) + self._loss_bg.apply(arr)

    def grad(self, arr):
        return self._loss_fg.grad(arr) + self._loss_bg.grad(arr)


@dataclass
class ReconstructionTechnique:
    ground_truth: pxt.NDArray
    op: xray.RayXRT
    regularizer: ProxFunc
    initialisation: pxt.NDArray
    diff_lip: float

    def run(self, stop_crit=RelError(eps=1e-3) | MaxIter(100), post_process_optres=None):
        data_fidelity = BhattInspiredLoss(self.op, self.ground_truth, b=b_param)
        pgd = PGD(data_fidelity, self.regularizer)
        pgd.fit(x0=self.initialisation, stop_crit=stop_crit, track_objective=True, tau=1 / self.diff_lip)
        alpha, _ = pgd.stats()
        alpha = alpha["x"]

        if post_process_optres is not None:
            alpha = post_process_optres(alpha)

        return self.op.adjoint(alpha)


def bunny_high():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__088.png")


def bunny_middle1():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__056.png")


def bunny_middle2():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__033.png")


def bunny_low():
    return 1 * iio.imread("../../3dtests/pngs/bunny_zres_100_reweighted__009.png")


ground_truth = bunny_high()

side = np.array(ground_truth.shape)
origin = 0.0
pitch = 1.0

angle = np.linspace(0, 2 * np.pi, num_n, endpoint=False)
t_max = pitch * side / 2  # 10% over ball radius
t_offset = np.linspace(-t_max[0], t_max[1], num_t, endpoint=True)

n = np.stack([np.cos(angle), np.sin(angle)], axis=1)
t = n[:, [1, 0]] * np.r_[-1, 1]  # <n, t> = 0

n_spec = np.broadcast_to(n.reshape(num_n, 1, 2), (num_n, num_t, 2))
t_spec = t.reshape(num_n, 1, 2) * t_offset.reshape(num_t, 1)
t_spec += pitch * side / 2

weighted_xrt = xray.RayXRT(
    dim_shape=ground_truth.shape,
    origin=origin,
    pitch=pitch,
    n_spec=n_spec.reshape(-1, 2),
    t_spec=t_spec.reshape(-1, 2),
)

unweighted_xrt = xray.RayXRT(
    dim_shape=ground_truth.shape,
    origin=origin,
    pitch=pitch,
    n_spec=n_spec.reshape(-1, 2),
    t_spec=t_spec.reshape(-1, 2),
)

lcav_low = ReconstructionTechnique(
    ground_truth=bunny_low(),
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(weighted_xrt.codim_shape),
    initialisation=np.zeros(weighted_xrt.codim_shape),
    diff_lip=diff_lip,
)

lcav_high = ReconstructionTechnique(
    ground_truth=bunny_high(),
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(weighted_xrt.codim_shape),
    initialisation=np.zeros(weighted_xrt.codim_shape),
    diff_lip=diff_lip,
)

lcav_middle1 = ReconstructionTechnique(
    ground_truth=bunny_middle1(),
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(weighted_xrt.codim_shape),
    initialisation=np.zeros(weighted_xrt.codim_shape),
    diff_lip=diff_lip,
)

lcav_middle2 = ReconstructionTechnique(
    ground_truth=bunny_middle2(),
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(weighted_xrt.codim_shape),
    initialisation=np.zeros(weighted_xrt.codim_shape),
    diff_lip=diff_lip,
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
