import imageio.v3 as iio
import numpy as np
import stltovoxel
import tqdm
from matplotlib import pyplot as plt


def read_stl_file(stl_path, png_path, npy_path, resolution=100):
    stltovoxel.convert_file(
        input_file_path=stl_path,
        output_file_path=png_path + ".png",
        resolution=resolution - 3,
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

    np.save(npy_path, array)


def show_voxels(npy_path, image_path):
    nut = np.load(npy_path)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.voxels(nut)
    fig.savefig(image_path)


if __name__ == "__main__":
    stl_path = "stls/Iron_Throne_Benchy.stl"
    png_path = "pngs/benchy_zres_200"
    npy_path = "npys/benchy_zres_200.npy"
    image_path = "images/benchy_zres200.png"
    resolution = 200

    read_stl_file(stl_path=stl_path, png_path=png_path, npy_path=npy_path, resolution=resolution)
    # show_voxels(npy_path=npy_path, image_path=image_path)
