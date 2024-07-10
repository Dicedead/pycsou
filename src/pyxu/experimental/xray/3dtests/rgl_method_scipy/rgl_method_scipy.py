import warnings
from dataclasses import dataclass

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import zarr

import pyxu.abc as pxa
import pyxu.info.ptype as pxt
import pyxu.operator.linop.xrt.ray as xray
from pyxu.abc import ProxFunc
from pyxu.experimental.xray.refraction import normalize, refract
from pyxu.operator import DiagonalOp, IdentityOp, PositiveOrthant
from pyxu.operator.linop.stencil.stencil import Stencil
from pyxu.opt.stop import MaxIter, RelError

warnings.filterwarnings("ignore")

dh = 0.95
dl = 0.9
num_n = 1500  # 3000
bin_size = 1
slm_pixels_height = 50  # 100
slm_pixels_width = 100  # 200
lambda_ = 40
diff_lip = 500_000
weighted_heavy = True
transmittance_ratio = 0.5 if weighted_heavy else 0.95


class BhattLossWeighted(pxa.DiffFunc):
    def __init__(
        self,
        xrt: xray.RayXRT,
        ground_truth: pxt.NDArray,
        mu1: pxt.Real,
        mu2: pxt.Real,
        dh: pxt.Real = dh,
        dl: pxt.Real = dl,
        z_weights: bool = True,
    ):
        super().__init__(dim_shape=xrt.codim_shape, codim_shape=(1,))
        weights = xp.sqrt(ground_truth.sum(axis=(0, 1)))
        weights = weights / xp.sqrt((weights**2).sum())
        weights = 1 - weights
        weighting = DiagonalOp(xp.ones(xrt.dim_shape) * weights) if z_weights else IdentityOp(dim_shape=xrt.dim_shape)
        fg_argshift = -dh * ground_truth
        bg_argshift = -dl * (1 - ground_truth)
        fg_constant = DiagonalOp(ground_truth)
        bg_constant = DiagonalOp(1 - ground_truth)
        pos_orth = PositiveOrthant(dim_shape=xrt.dim_shape)
        fg_posort = pos_orth.argshift(fg_argshift).moreau_envelope(mu1)
        bg_posort = pos_orth.argshift(bg_argshift).argscale(-1).moreau_envelope(mu1)
        nb_ones = int(xp.ceil(ground_truth.shape[-1] / slm_pixels_height))
        s = Stencil(dim_shape=xrt.dim_shape, kernel=[xp.r_[1], xp.r_[1], xp.ones(nb_ones)], center=[0, 0, nb_ones - 1])
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
    z_weights: bool

    def run(self, stop_crit=RelError(eps=1e-3) | MaxIter(200), post_process_optres=None, mu1=10, mu2=10, n_iter=50):
        print("########### First and last epoch")
        alpha, hist = self.__run_epoch(
            self.initialisation, mu1=mu1, mu2=mu2, stop_crit=RelError(eps=1e-3) | MaxIter(n_iter)
        )

        alpha_copy = alpha.copy()

        if post_process_optres is not None:
            alpha = post_process_optres(alpha)

        alpha = alpha.clip(0, None)
        return alpha_copy, self.op.adjoint(alpha), hist

    def __run_epoch(self, x0: pxt.NDArray, mu1: pxt.Real, mu2: pxt.Real, stop_crit: pxa.StoppingCriterion):
        x0_init_size = x0.shape
        x0 = x0.flatten()
        loss = BhattLossWeighted(self.op, self.ground_truth, mu1=mu1, mu2=mu2, z_weights=z_weights)
        hist = np.r_[0.0]
        res = sp.optimize.minimize(
            fun=loss,
            x0=x0,
            jac=loss.grad,
            method="L-BFGS-B",
            options={"maxiter": 40, "iprint": 1},
        )
        hist = {"Memorize[objective_func]": hist}
        return res.x.reshape(x0_init_size), hist


def ellipsis(side_a, num_a, side_b, num_b):
    side_a = side_a / 2
    side_b = side_b / 2
    x, y = xp.meshgrid(xp.linspace(-side_a, side_a, num_a), xp.linspace(-side_b, side_b, num_b))
    ground_truth = 1 * ((x / side_a) ** 2 + (y / side_b) ** 2 <= 1)
    return ground_truth


def absorption_coeff(sides, transmittance_ratio):
    assert len(sides) == 3
    sides = pitch * sides
    return -xp.log(transmittance_ratio) * xp.sum(sides) / (3 * xp.prod(sides))


def benchy_padded(path="../npys/benchy_padded_150.zarr"):
    return 1 * zarr.load(path)


def nut_padded(path="../npys/nut_padded_150.zarr"):
    return 1 * zarr.load(path)


def bunny_padded(path="../npys/bunny_zres_150_padded.npy"):
    return 1 * xp.load(path)


print("Loading ground truth...")
optimize_save = False
refraction = True
weighted = True
z_weights = True
gpu = False
gpu = gpu and optimize_save
xp = cp if gpu else np
ground_truth = bunny_padded()
chosen_gt = "bunny_padded"
ground_truth = ground_truth if not gpu else cp.array(ground_truth)
chosen_gt = chosen_gt + "_weighted" if weighted else chosen_gt
chosen_gt = chosen_gt + "_refracted" if refraction else chosen_gt
chosen_gt = chosen_gt + "_no_z_weights" if not z_weights else chosen_gt

origin = 0
pitch = xp.array([1.0, 1.0, 1.0])

print("Creating rays...")
num_heights = slm_pixels_height // bin_size
num_offsets = slm_pixels_width // bin_size

side = xp.array(ground_truth.shape)
max_height = ground_truth.shape[-1]
angle = xp.linspace(0, 2 * xp.pi, num_n, endpoint=False)
heights = xp.linspace(0.0000001, max_height, num_heights, endpoint=False)
t_max = pitch * side / 2
t_max = t_max[:-1]
max_t_max = xp.max(t_max)
t_offset = xp.linspace(-max_t_max, max_t_max, num_offsets, endpoint=False)

n = xp.stack([xp.cos(angle), xp.sin(angle), xp.zeros(num_n)], axis=1)
t = n[:, [1, 0]] * xp.r_[-1, 1]  # <n, t> = 0
t = xp.tile(t, num_heights).reshape(-1, 2)
heights = xp.tile(heights, num_n).reshape(-1, 1)
t = xp.hstack([t, heights]).reshape(num_n * num_heights, 1, 3)

n_spec = xp.broadcast_to(n.reshape(num_n, 1, 3), (num_n, num_offsets, 3))  # (N_angle, N_offset, 3)
n_spec = xp.tile(n_spec, num_heights).reshape((num_n * num_heights, num_offsets, 3))
t_spec = t * t_offset.reshape(num_offsets, 1)
t_spec[:, :, -1] = t[:, :, -1]
t_spec += xp.r_[pitch[:2], 0] * xp.array(ground_truth.shape) / 2

n_spec = n_spec.reshape(-1, 3)
t_spec = t_spec.reshape(-1, 3)

if refraction:
    external_diameter = 5 * 2 * max(side[0], side[1]) + 10

    c_spec = [1, external_diameter, 0, side[-1], 0, side[-1]]
    r_spec = [1, 1.4, 1.45]

    t_spec -= 3 * external_diameter * n_spec

    n_spec, t_spec = refract(n_spec.reshape(-1, 3), t_spec.reshape(-1, 3), r_spec, c_spec)
    n_spec = n_spec[~xp.isnan(n_spec.sum(axis=-1))]
    t_spec = t_spec[~xp.isnan(t_spec.sum(axis=-1))]
    n_spec = normalize(n_spec)

if weighted:
    w_spec = absorption_coeff(side, transmittance_ratio) * ellipsis(
        int(pitch[1] * side[1]), int(side[1]), int(pitch[0] * side[0]), int(side[0])
    )
    w_spec = w_spec[:, :, xp.newaxis] * xp.ones(shape=(1, 1, int(side[-1])))

print("Building operator...")
xrt = (
    xray.RayXRT(
        dim_shape=ground_truth.shape,
        origin=origin,
        pitch=pitch,
        n_spec=n_spec,
        t_spec=t_spec,
    )
    if not weighted
    else xray.RayWXRT(
        dim_shape=ground_truth.shape, origin=origin, pitch=pitch, n_spec=n_spec, t_spec=t_spec, w_spec=w_spec
    )
)

print("Diagnostic plot...")
# fig = xrt.diagnostic_plot()
# fig.savefig("./diag.png")

bhatt = ReconstructionTechnique(
    ground_truth=ground_truth,
    op=xrt,
    regularizer=PositiveOrthant(xrt.codim_shape),
    initialisation=xp.zeros(xrt.codim_shape),
    diff_lip=diff_lip,
    z_weights=z_weights,
)


def threshold_processing_2_colors(image):
    thresh = (dh + dl) / 2
    res = image.copy()
    res[image < thresh] = 0
    res[image >= thresh] = 1
    return res


def threshold_processing_2_colors_bis(image):
    thresh = dh
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


def zero_order_interp(img: xp.ndarray, epsilon=1e-6):
    output = img.copy()
    nonzero_planes = np.argwhere(img.sum(axis=(0, 1)) > epsilon).flatten()
    nonzero_length = len(nonzero_planes) - 1
    idx = 0
    while idx < nonzero_length:
        z = nonzero_planes[idx]
        z_plus_1 = nonzero_planes[idx + 1]
        output[:, :, z:z_plus_1] = output[:, :, z, xp.newaxis]
        idx += 1

    return output, nonzero_planes


def plot_opt_history(hist, hist_file_title):
    plt.plot(hist["Memorize[objective_func]"])
    plt.savefig(hist_file_title, dpi=500)


def run():
    global ground_truth

    save_file = f"alphas/alpha_{chosen_gt}.zarr"
    hist_file = f"hist/hist_{chosen_gt}.zarr"

    if optimize_save:
        alpha, img, hist = bhatt.run()

        if gpu:
            alpha = alpha.get()
            img = img.get()
            ground_truth = ground_truth.get()

        zarr.save(save_file, alpha)
        zarr.save(hist_file, hist)
    else:
        alpha = zarr.load(save_file)
        hist = zarr.load(hist_file)
        img = bhatt.op.adjoint(alpha.clip(0, None))

    img, nonzero_planes = zero_order_interp(img)

    show_projection_against_gt(
        0, img, ground_truth, f"{chosen_gt} x projection", f"results/{chosen_gt}/x_prog_normalized.png"
    )
    show_projection_against_gt(
        0, img, ground_truth, f"{chosen_gt} x projection", f"results/{chosen_gt}/x_prog.png", normalize=False
    )
    show_projection_against_gt(
        0,
        threshold_processing_2_colors_bis(img),
        ground_truth,
        f"{chosen_gt} x projection",
        f"results/{chosen_gt}/x_prog_binarized.png",
        normalize=False,
    )

    show_projection_against_gt(
        1, img, ground_truth, f"{chosen_gt} y projection", f"results/{chosen_gt}/y_prog_normalized.png"
    )
    show_projection_against_gt(
        1, img, ground_truth, f"{chosen_gt} y projection", f"results/{chosen_gt}/y_prog.png", normalize=False
    )
    show_projection_against_gt(
        1,
        threshold_processing_2_colors_bis(img),
        ground_truth,
        f"{chosen_gt} y projection",
        f"results/{chosen_gt}/y_prog_binarized.png",
        normalize=False,
    )

    show_projection_against_gt(
        2, img, ground_truth, f"{chosen_gt} z projection", f"results/{chosen_gt}/z_prog_normalized.png"
    )
    show_projection_against_gt(
        2, img, ground_truth, f"{chosen_gt} z projection", f"results/{chosen_gt}/z_prog.png", normalize=False
    )
    show_projection_against_gt(
        2,
        threshold_processing_2_colors_bis(img),
        ground_truth,
        f"{chosen_gt} z projection",
        f"results/{chosen_gt}/z_prog_binarized.png",
        normalize=False,
    )

    low_index = (
        int(np.ceil(ground_truth.shape[-1] * 0.25))
        if len(nonzero_planes) == 0
        else nonzero_planes[int(np.ceil(nonzero_planes.shape[-1] * 0.25))]
    )
    high_index = (
        int(np.ceil(ground_truth.shape[-1] * 0.75))
        if len(nonzero_planes) == 0
        else nonzero_planes[int(np.floor(nonzero_planes.shape[-1] * 0.75))]
    )

    plot_four_images(
        [ground_truth[:, :, low_index], img[:, :, low_index], ground_truth[:, :, high_index], img[:, :, high_index]],
        subtitles=["Low GT", "Low data", "High GT", "High data"],
        main_title=f"{chosen_gt} slices",
        file_title=f"results/{chosen_gt}/slices_3col.png",
        processing=[lambda x: x, threshold_processing_3_colors, lambda x: x, threshold_processing_3_colors],
    )

    plot_four_images(
        [ground_truth[:, :, low_index], img[:, :, low_index], ground_truth[:, :, high_index], img[:, :, high_index]],
        subtitles=["Low GT", "Low data", "High GT", "High data"],
        main_title=f"{chosen_gt} slices",
        file_title=f"results/{chosen_gt}/slices_2col.png",
        processing=[lambda x: x, threshold_processing_2_colors, lambda x: x, threshold_processing_2_colors],
    )

    plot_four_images(
        [ground_truth[:, :, low_index], img[:, :, low_index], ground_truth[:, :, high_index], img[:, :, high_index]],
        subtitles=["Low GT", "Low data", "High GT", "High data"],
        main_title=f"{chosen_gt} slices",
        file_title=f"results/{chosen_gt}/slices_energy.png",
    )

    hist_file_title = f"results/{chosen_gt}/history.png"
    plot_opt_history(hist, hist_file_title)


if __name__ == "__main__":
    run()
