import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import zarr

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.linop.xrt.ray as xray
from pyxu.abc import ProxFunc
from pyxu.operator import DiagonalOp, PositiveOrthant
from pyxu.operator.linop.stencil.stencil import Stencil
from pyxu.opt.solver import PGD
from pyxu.opt.stop import MaxIter, RelError

# from pyxu.experimental.xray.refraction import refract, normalize

warnings.filterwarnings("ignore")

dh = 0.9
dl = 0.5
num_n = 3000  # 3000
bin_size = 1
slm_pixels_height = 50  # 100
slm_pixels_width = 100  # 200
lambda_ = 4
diff_lip = 0.5


class BhattLossWeighted(pxa.DiffFunc):
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
        weights = np.sqrt(ground_truth.sum(axis=(0, 1)))
        weights = weights / np.sqrt((weights**2).sum())
        weights = 1 - weights
        weighting = DiagonalOp(np.ones(xrt.dim_shape) * weights)
        fg_argshift = -dh * ground_truth
        bg_argshift = -dl * (1 - ground_truth)
        fg_constant = DiagonalOp(ground_truth)
        bg_constant = DiagonalOp(1 - ground_truth)
        pos_orth = PositiveOrthant(dim_shape=xrt.dim_shape)
        fg_posort = pos_orth.argshift(fg_argshift).moreau_envelope(mu1)
        bg_posort = pos_orth.argshift(bg_argshift).argscale(-1).moreau_envelope(mu1)
        nb_ones = int(np.ceil(ground_truth.shape[-1] / slm_pixels_height))
        s = Stencil(dim_shape=xrt.dim_shape, kernel=[np.r_[1], np.r_[1], np.ones(nb_ones)], center=[0, 0, nb_ones - 1])
        self._inner_op = (fg_posort * fg_constant + bg_posort * bg_constant) * weighting * s * xrt.T
        self._xrt = xrt
        self._reg = lambda_ * PositiveOrthant(xrt.codim_shape).moreau_envelope(mu2)

    def apply(self, arr):
        return self._inner_op(arr) + self._reg(arr)

    def grad(self, arr):
        return self._inner_op.grad(arr) + self._reg.grad(arr)


@dataclass
class ReconstructionTechnique:
    ground_truth: pxt.NDArray
    op: xray.RayXRT
    regularizer: ProxFunc
    initialisation: pxt.NDArray
    diff_lip: float

    def run(self, stop_crit=RelError(eps=1e-3) | MaxIter(200), post_process_optres=None, mu1=10, mu2=10):
        alpha = self.__run_epoch(self.initialisation, mu1=mu1, mu2=mu2, stop_crit=RelError(eps=1e-3) | MaxIter(50))
        # alpha = self.__run_epoch(alpha, mu1=mu1 / 2, mu2=mu2, stop_crit=RelError(eps=1e-3) | MaxIter(50))
        # alpha = self.__run_epoch(alpha, mu1=mu1 / 2, mu2=mu2 / 2, stop_crit=RelError(eps=1e-3) | MaxIter(50))
        # alpha = self.__run_epoch(alpha, mu1=mu1 / 4, mu2=mu2 / 2, stop_crit=RelError(eps=1e-5) | MaxIter(50))
        # alpha = self.__run_epoch(alpha, mu1=mu1 / 4, mu2=mu2 / 4, stop_crit=RelError(eps=1e-5) | MaxIter(50))
        # alpha = self.__run_epoch(alpha, mu1=mu1 / 8, mu2=mu2 / 4, stop_crit=RelError(eps=5e-6) | MaxIter(50))
        # alpha = self.__run_epoch(alpha, mu1=mu1 / 8, mu2=mu2 / 8, stop_crit=RelError(eps=5e-6) | MaxIter(50))

        alpha_copy = alpha.copy()

        if post_process_optres is not None:
            alpha = post_process_optres(alpha)

        alpha = alpha.clip(0, None)
        return alpha_copy, self.op.adjoint(alpha)

    def __run_epoch(self, x0: pxt.NDArray, mu1: pxt.Real, mu2: pxt.Real, stop_crit: pxa.StoppingCriterion):
        loss = BhattLossWeighted(self.op, self.ground_truth, mu1=mu1, mu2=mu2)
        pgd = PGD(loss)
        pgd.fit(x0=x0, stop_crit=stop_crit, track_objective=True, tau=1 / self.diff_lip)
        alpha, _ = pgd.stats()
        return alpha["x"]


def ellipsis(side_a, num_a, side_b, num_b):
    side_a = side_a / 2
    side_b = side_b / 2
    x, y = np.meshgrid(np.linspace(-side_a, side_a, num_a), np.linspace(-side_b, side_b, num_b))
    ground_truth = 1 * ((x / side_a) ** 2 + (y / side_b) ** 2 <= 1)
    return ground_truth


def absorption_coeff(sides, transmittance_ratio):
    assert len(sides) == 2
    sides = pitch * sides
    return -np.log(transmittance_ratio) * np.sum(sides) / (2 * np.prod(sides))


def nut(path="../npys/nut_zres100.npy"):
    return 1 * np.load(path)


def bunny(path="../npys/bunny_zres_100.npy"):
    return 1 * np.load(path)


def bunny_reweighted(path="../npys/bunny_zres_100_reweighted.npy"):
    return 1 * np.load(path)


def benchy(path="../npys/benchy_zres_200.npy"):
    return 1 * np.load(path)


def bunny_padded(path="../npys/bunny_zres_150_padded.npy"):
    return 1 * np.load(path)


print("Loading ground truth...")
ground_truth = bunny_padded()
chosen_gt = "bunny_padded"
refraction = False
optimize_save = False

cylinder_inner_radius = 155  # 15.5e-3
cylinder_outer_radius = 165  # 16.5e-3
cylinder_max_height = 0.0105216 * 3 * 1e4  # 0.0105216 * 3
cylinder_min_height = 0
assert cylinder_inner_radius < cylinder_outer_radius

origin = 0
vox_side = 1  # 13.7e-5
max_height = cylinder_max_height
max_offset = cylinder_outer_radius / 10
pitch = vox_side * np.array([1.0, 1.0, 1.0])

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
t_spec += np.r_[pitch[:2], 0] * np.array(ground_truth.shape) / 2

n_spec = n_spec.reshape(-1, 3)
t_spec = t_spec.reshape(-1, 3)

# if refraction:
#     possible_folders = [f"{x}_with_refraction" for x in possible_folders]
#     folder = possible_folders[idx_chosen]
#
#     external_diameter = 5 * 2 * max(t_max) + 10
#
#     c_spec = [1, external_diameter, 0, 10, 0, 10]
#     r_spec = [1, 1.4, 1.45]
#
#     t_spec -= 3 * external_diameter * n_spec
#
#     n_spec, t_spec = refract(
#         np.pad(n_spec.reshape(-1, 2), pad_width=[(0, 0), (0, 1)], constant_values=0),
#         np.pad(t_spec.reshape(-1, 2), pad_width=[(0, 0), (0, 1)], constant_values=5),
#         r_spec,
#         c_spec,
#     )
#     n_spec = n_spec[~np.isnan(n_spec)]
#     t_spec = t_spec[~np.isnan(t_spec)]
#     n_spec = normalize(n_spec.reshape(-1, 3)[:, :2])
#     t_spec = t_spec.reshape(-1, 3)[:, :2]

# w_spec = absorption_coeff(side, transmittance_ratio) * ellipsis(pitch * side[1], side[1], pitch * side[0], side[0])

print("Building operator...")
unweighted_xrt = xray.RayXRT(
    dim_shape=ground_truth.shape,
    origin=origin,
    pitch=pitch,
    n_spec=n_spec,
    t_spec=t_spec,
)

print("Diagnostic plot...")
# fig = unweighted_xrt.diagnostic_plot()
# fig.savefig("./diag.png")

bhatt = ReconstructionTechnique(
    ground_truth=ground_truth,
    op=unweighted_xrt,
    regularizer=PositiveOrthant(unweighted_xrt.codim_shape),
    initialisation=np.zeros(unweighted_xrt.codim_shape),
    diff_lip=diff_lip,
)


def threshold_processing_2_colors(image):
    thresh = (dh + dl) / 2
    res = image.copy()
    res[image < thresh] = 0
    res[image >= thresh] = 1
    return res


def threshold_processing_3_colors(image):
    res = image.copy()
    res[image < dl] = 0
    res[(image >= dl) & (image < dh)] = 0.5
    res[image > dh] = 1
    return res


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


def show_projection_against_gt(ax, data, ground_truth, main_title, file_title, processing=None, normalize=True):
    if normalize:
        proj1 = np.sum(data, axis=ax) / data.shape[ax]
        proj2 = np.sum(ground_truth, axis=ax) / ground_truth.shape[ax]
    else:
        proj1 = np.sum(data, axis=ax)
        proj2 = np.sum(ground_truth, axis=ax)

    if processing is None:
        processing = [lambda x: x] * 2

    subtitles = ["Ground truth", "Data"]
    imgs = [proj2, proj1]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax_flat = ax.flatten()
    fig.suptitle(main_title, fontsize=16)

    for idx, a in enumerate(ax_flat):
        if normalize:
            im = a.imshow(processing[idx](imgs[idx]), cmap="Greys", vmin=0, vmax=1)
        else:
            im = a.imshow(processing[idx](imgs[idx]), cmap="Greys")
        a.set_title(subtitles[idx])
        plt.colorbar(im, ax=a)

    fig.savefig(file_title, dpi=500)
    plt.close(fig)


def zero_order_interp(img: np.ndarray, epsilon=1e-6):
    output = img.copy()
    nonzero_planes = np.argwhere(img.sum(axis=(0, 1)) > epsilon).flatten()
    nonzero_length = len(nonzero_planes) - 1
    idx = 0
    while idx < nonzero_length:
        z = nonzero_planes[idx]
        z_plus_1 = nonzero_planes[idx + 1]
        output[:, :, z:z_plus_1] = output[:, :, z, np.newaxis]
        idx += 1

    return output, nonzero_planes


def run():
    save_file = f"alphas/alpha_{chosen_gt}.zarr"
    if optimize_save:
        alpha, img = bhatt.run()
        zarr.save(save_file, alpha)
    else:
        alpha = zarr.load(save_file)
        img = bhatt.op.adjoint(alpha)

    img, nonzero_planes = zero_order_interp(img)

    show_projection_against_gt(
        0, img, ground_truth, f"{chosen_gt} x projection", f"results/{chosen_gt}_x_prog_normalized.png"
    )
    show_projection_against_gt(
        0, img, ground_truth, f"{chosen_gt} x projection", f"results/{chosen_gt}_x_prog.png", normalize=False
    )

    show_projection_against_gt(
        1, img, ground_truth, f"{chosen_gt} y projection", f"results/{chosen_gt}_y_prog_normalized.png"
    )
    show_projection_against_gt(
        1, img, ground_truth, f"{chosen_gt} y projection", f"results/{chosen_gt}_y_prog.png", normalize=False
    )

    show_projection_against_gt(
        2, img, ground_truth, f"{chosen_gt} z projection", f"results/{chosen_gt}_z_prog_normalized.png"
    )
    show_projection_against_gt(
        2, img, ground_truth, f"{chosen_gt} z projection", f"results/{chosen_gt}_z_prog.png", normalize=False
    )

    low_index = nonzero_planes[int(np.ceil(nonzero_planes.shape[-1] * 0.25))]
    high_index = nonzero_planes[int(np.floor(nonzero_planes.shape[-1] * 0.75))]

    plot_four_images(
        [ground_truth[:, :, low_index], img[:, :, low_index], ground_truth[:, :, high_index], img[:, :, high_index]],
        subtitles=["Low GT", "Low data", "High GT", "High data"],
        main_title=f"{chosen_gt} slices",
        file_title=f"results/{chosen_gt}_slices_3col.png",
        processing=[lambda x: x, threshold_processing_3_colors, lambda x: x, threshold_processing_3_colors],
    )

    plot_four_images(
        [ground_truth[:, :, low_index], img[:, :, low_index], ground_truth[:, :, high_index], img[:, :, high_index]],
        subtitles=["Low GT", "Low data", "High GT", "High data"],
        main_title=f"{chosen_gt} slices",
        file_title=f"results/{chosen_gt}_slices_2col.png",
        processing=[lambda x: x, threshold_processing_2_colors, lambda x: x, threshold_processing_2_colors],
    )

    plot_four_images(
        [ground_truth[:, :, low_index], img[:, :, low_index], ground_truth[:, :, high_index], img[:, :, high_index]],
        subtitles=["Low GT", "Low data", "High GT", "High data"],
        main_title=f"{chosen_gt} slices",
        file_title=f"results/{chosen_gt}_slices_energy.png",
    )


if __name__ == "__main__":
    run()
