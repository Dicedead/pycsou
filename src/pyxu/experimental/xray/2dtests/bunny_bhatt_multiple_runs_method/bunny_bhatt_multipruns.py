import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.linop.xrt.ray as xray
from pyxu.abc import ProxFunc
from pyxu.operator import DiagonalOp, PositiveOrthant
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError

warnings.filterwarnings("ignore")

dh = 0.9
dl = 0.5
num_n = 1000
num_t = 350
lambda_ = 4


class BhattLoss(pxa.DiffFunc):
    def __init__(
        self,
        xrt: xray.RayXRT,
        ground_truth: pxt.NDArray,
        mu1: pxt.Real,
        mu2: pxt.Real,
        dh: pxt.Real = dh,
        dl: pxt.Real = dl,
    ):
        super().__init__(dim_shape=xrt.codim_shape, codim_shape=(1,))
        self._mask = ground_truth > 0
        fg_argshift = -dh * ground_truth
        bg_argshift = -dl * (1 - ground_truth)  # TODO if we reweight later, this needs to change
        fg_constant = DiagonalOp(ground_truth)
        bg_constant = DiagonalOp(1 - ground_truth)
        fg_posort = PositiveOrthant(dim_shape=xrt.dim_shape).argshift(fg_argshift).moreau_envelope(mu1)
        bg_posort = PositiveOrthant(dim_shape=xrt.dim_shape).argshift(bg_argshift).argscale(-1).moreau_envelope(mu1)
        self._loss_fg = fg_posort * (fg_constant * xrt.T)
        self._loss_bg = bg_posort * (bg_constant * xrt.T)
        self._reg = lambda_ * PositiveOrthant(unweighted_xrt.codim_shape).moreau_envelope(mu2)

    def apply(self, arr):
        return self._loss_fg.apply(arr) + self._loss_bg.apply(arr) + self._reg(arr)
        # TODO naive implem because XRT.adj is applied twice => can be optimized if needed

    def grad(self, arr):
        return self._loss_fg.grad(arr) + self._loss_bg.grad(arr) + self._reg.grad(arr)


@dataclass
class ReconstructionTechnique:
    ground_truth: pxt.NDArray
    op: xray.RayXRT
    regularizer: ProxFunc
    initialisation: pxt.NDArray
    diff_lip: float

    def run(self, stop_crit=RelError(eps=1e-3) | MaxIter(200), post_process_optres=None, mu1=10, mu2=10):
        alpha = self.__run_epoch(self.initialisation, mu1=mu1, mu2=mu2, stop_crit=RelError(eps=1e-3) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 2, mu2=mu2, stop_crit=RelError(eps=1e-3) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 2, mu2=mu2 / 2, stop_crit=RelError(eps=1e-3) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 4, mu2=mu2 / 2, stop_crit=RelError(eps=1e-5) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 4, mu2=mu2 / 4, stop_crit=RelError(eps=1e-5) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 8, mu2=mu2 / 4, stop_crit=RelError(eps=5e-6) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 8, mu2=mu2 / 8, stop_crit=RelError(eps=5e-6) | MaxIter(200))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 16, mu2=mu2 / 8, stop_crit=RelError(eps=1e-6) | MaxIter(300))
        alpha = self.__run_epoch(alpha, mu1=mu1 / 16, mu2=mu2 / 16, stop_crit=RelError(eps=1e-6) | MaxIter(300))

        if post_process_optres is not None:
            alpha = post_process_optres(alpha)

        tmp = len(alpha[alpha > 0]) / len(alpha)
        print(tmp)
        alpha = alpha.clip(0, None)
        return self.op.adjoint(alpha)

    def __run_epoch(self, x0: pxt.NDArray, mu1: pxt.Real, mu2: pxt.Real, stop_crit: pxa.StoppingCriterion):
        loss = BhattLoss(self.op, self.ground_truth, mu1=mu1, mu2=mu2)
        pgd = PGD(loss)
        pgd.fit(x0=x0, stop_crit=stop_crit, track_objective=True, tau=1 / self.diff_lip)
        alpha, _ = pgd.stats()
        return alpha["x"]


bunny_low_res = bunny_more_res = np.load("../../3dtests/npys/bunny_zres_100.npy")


def bunny_high():
    return bunny_low_res[:, :, 88]


def bunny_middle1():
    return bunny_low_res[:, :, 56]


def bunny_middle2():
    return bunny_low_res[:, :, 33]


def bunny_low():
    return bunny_low_res[:, :, 9]


bunny_more_res = np.load("../../3dtests/npys/bunny_zres_200.npy")


def bunny_high_more_res():
    return bunny_more_res[:, :, 185]


def bunny_middle1_more_res():
    return bunny_more_res[:, :, 63]


def bunny_middle2_more_res():
    return bunny_more_res[:, :, 104]


def bunny_low_more_res():
    return bunny_more_res[:, :, 12]


padded_bunny = np.load("../../3dtests/npys/bunny_zres_150_padded.npy")


def bunny_high_padded():
    return padded_bunny[:, :, 133]


def bunny_middle1_padded():
    return padded_bunny[:, :, 74]


def bunny_middle2_padded():
    return padded_bunny[:, :, 84]


def bunny_low_padded():
    return padded_bunny[:, :, 9]


low_res = [bunny_low(), bunny_middle1(), bunny_middle2(), bunny_high()]
higher_res = [bunny_low_more_res(), bunny_middle1_more_res(), bunny_middle2_more_res(), bunny_high_more_res()]
padded = [bunny_low_padded(), bunny_middle1_padded(), bunny_middle2_padded(), bunny_high_padded()]
idx_chosen = 2

possible_gts = [low_res, higher_res, padded]
possible_folders = ["res_100", "res_200", "res_150_padded"]

ground_truth = possible_gts[idx_chosen]
folder = possible_folders[idx_chosen]

side = np.array(ground_truth[0].shape)
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

unweighted_xrt = xray.RayXRT(
    dim_shape=ground_truth[0].shape,
    origin=origin,
    pitch=pitch,
    n_spec=n_spec.reshape(-1, 2),
    t_spec=t_spec.reshape(-1, 2),
)

lcav_low = ReconstructionTechnique(
    ground_truth=ground_truth[0],
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(unweighted_xrt.codim_shape),
    initialisation=np.zeros(unweighted_xrt.codim_shape),
    diff_lip=40000,
)

lcav_high = ReconstructionTechnique(
    ground_truth=ground_truth[-1],
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(unweighted_xrt.codim_shape),
    initialisation=np.zeros(unweighted_xrt.codim_shape),
    diff_lip=40000,
)

lcav_middle1 = ReconstructionTechnique(
    ground_truth=ground_truth[1],
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(unweighted_xrt.codim_shape),
    initialisation=np.zeros(unweighted_xrt.codim_shape),
    diff_lip=40000,
)

lcav_middle2 = ReconstructionTechnique(
    ground_truth=ground_truth[2],
    op=unweighted_xrt,
    regularizer=lambda_ * PositiveOrthant(unweighted_xrt.codim_shape),
    initialisation=np.zeros(unweighted_xrt.codim_shape),
    diff_lip=40000,
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


def threshold_processing_1(image):
    thresh = (dh + dl) / 2
    res = image.copy()
    res[image < thresh] = 0
    res[image >= thresh] = 1
    return res


def threshold_processing_2(image):
    res = image.copy()
    res[image < dl] = 0
    res[(image >= dl) & (image < dh)] = 0.5
    res[image > dh] = 1
    return res


def run_both_middle(folder=""):
    lcav_middle1_img = lcav_middle1.run()
    lcav_middle2_img = lcav_middle2.run()

    plot_four_images(
        [ground_truth[1], lcav_middle1_img, ground_truth[2], lcav_middle2_img],
        ["GT mid 1", "Mid 1", "GT mid 2", "Mid 2"],
        "No thresholding",
        f"results/{folder}/no_thresholding_part2.png",
    )

    plot_four_images(
        [ground_truth[1], lcav_middle1_img, ground_truth[2], lcav_middle2_img],
        ["GT mid 1", "Mid 1", "GT mid 2", "Mid 2"],
        "With thresholding - 2 colors",
        f"results/{folder}/with_thresholding_2_colors_part2.png",
        processing=[lambda x: x, threshold_processing_1, lambda x: x, threshold_processing_1],
    )

    plot_four_images(
        [ground_truth[1], lcav_middle1_img, ground_truth[2], lcav_middle2_img],
        ["GT mid 1", "Mid 1", "GT mid 2", "Mid 2"],
        "With thresholding - 3 colors",
        f"results/{folder}/with_thresholding_3_colors_part2.png",
        processing=[lambda x: x, threshold_processing_2, lambda x: x, threshold_processing_2],
    )


def run_high_low(folder=""):
    lcav_low_img = lcav_low.run()
    lcav_high_img = lcav_high.run()

    plot_four_images(
        [ground_truth[0], lcav_low_img, ground_truth[-1], lcav_high_img],
        ["GT low", "Low", "GT high", "High"],
        "No thresholding",
        f"results/{folder}/no_thresholding_part1.png",
    )

    plot_four_images(
        [ground_truth[0], lcav_low_img, ground_truth[-1], lcav_high_img],
        ["GT low", "Low", "GT high", "High"],
        "With thresholding - 2 colors",
        f"results/{folder}/with_thresholding_2_colors_part1.png",
        processing=[lambda x: x, threshold_processing_1, lambda x: x, threshold_processing_1],
    )

    plot_four_images(
        [ground_truth[0], lcav_low_img, ground_truth[-1], lcav_high_img],
        ["GT low", "Low", "GT high", "High"],
        "With thresholding - 3 colors",
        f"results/{folder}/with_thresholding_3_colors_part1.png",
        processing=[lambda x: x, threshold_processing_2, lambda x: x, threshold_processing_2],
    )


if __name__ == "__main__":
    # run_high_low(folder)
    run_both_middle(folder)
