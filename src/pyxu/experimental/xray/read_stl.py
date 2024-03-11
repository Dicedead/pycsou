import imageio.v3 as iio
import numpy as np
import stltovoxel
import tqdm
from matplotlib import pyplot as plt


def read_stl_file(stl_path, png_path, npy_path, resolution=100):
    stltovoxel.convert_file(
        input_file_path=stl_path,
        output_file_path=png_path + ".png",
        resolution=resolution - 2,
        voxel_size=None,
        pad=1,
        parallel=False,
    )

    arrays = []
    num_digits = 1 + int(np.floor(np.log10(resolution)))
    num_digits = f"0{num_digits}"
    for i in tqdm.trange(resolution):
        arrays.append(iio.imread(f"{png_path}_{i:{num_digits}}.png"))
    array = np.stack(arrays, axis=2)

    summation = array.sum(axis=(1, 2), keepdims=True)
    array = array / np.maximum(np.ones_like(summation), summation)

    np.save(npy_path, array)


def show_voxels(npy_path, image_path):
    nut = np.load(npy_path)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.voxels(nut)
    fig.savefig(image_path)


if __name__ == "__main__":
    stl_path = "stls/Bunny-LowPoly.stl"
    png_path = "pngs/bunny_zres_100_reweighted_"
    npy_path = "npys/bunny_zres_100_reweighted.npy"
    image_path = "images/bunny_zres_100_reweighted.png"
    resolution = 100

    read_stl_file(stl_path=stl_path, png_path=png_path, npy_path=npy_path, resolution=resolution)
    # show_voxels(npy_path=npy_path, image_path=image_path)
