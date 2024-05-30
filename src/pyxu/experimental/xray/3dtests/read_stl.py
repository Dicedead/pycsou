import imageio.v3 as iio
import numpy as np
import stltovoxel
import tqdm
import zarr
from matplotlib import pyplot as plt


def read_stl_file(stl_path, png_path, npy_path, resolution=100, reweighting=True, pad_width=0):
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
        tmp = 1 * iio.imread(f"{png_path}_{i:{num_digits}}.png")
        if pad_width > 0:
            tmp = np.pad(tmp, pad_width)
        arrays.append(tmp)
    array = np.stack(arrays, axis=2)

    if reweighting:
        summation = array.sum(axis=(1, 2), keepdims=True)
        array = array / np.maximum(np.ones_like(summation), np.sqrt(summation))

    zarr.save(npy_path, array)


def show_voxels(npy_path, image_path):
    nut = np.load(npy_path)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.voxels(nut)
    fig.savefig(image_path)


if __name__ == "__main__":
    stl_path = "stls/Iron_Throne_Benchy.stl"
    png_path = "pngs/benchy_padded_150"
    npy_path = "npys/benchy_padded_150.zarr"
    image_path = "images/benchy_padded_150.png"
    resolution = 150
    reweight = False

    read_stl_file(
        stl_path=stl_path,
        png_path=png_path,
        npy_path=npy_path,
        resolution=resolution,
        reweighting=reweight,
        pad_width=30,
    )
    # show_voxels(npy_path=npy_path, image_path=image_path)
