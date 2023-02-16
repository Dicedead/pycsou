import matplotlib.pyplot as plt
import numpy as np

from pycsou.operator import Stencil

__WALL_DISTANCE = 7


def c_mat(c, mat):
    return (mat.T * c).T


def normalize(vs):
    vs = vs.T
    return (vs / np.linalg.norm(vs, axis=0)).T


def snell_law(incident_ray, normal, n1=1.000277, n2=1.5):
    # refraction indices from Mitsuba test

    k = incident_ray  # shorthand
    n = normal  # shorthand

    cross_prod = np.cross(n, k)
    frac = n1 / n2

    q1 = frac * np.cross(n, -cross_prod)
    c = np.sqrt(1 - (frac * np.linalg.norm(cross_prod, axis=1)) ** 2)

    return q1 - c_mat(c, n)


def bin_grid(vectors, resolution):
    def get_bin(val, min_val, width):
        return resolution - 1 - int(np.clip(np.floor_divide(resolution * (val - min_val), width), 0, resolution - 1))

    vectors = vectors.T

    xs = vectors[0]
    ys = vectors[2]

    min_y = np.min(ys)
    max_y = np.max(ys)

    min_x = np.min(xs)
    max_x = np.max(xs)

    y_width = max_y - min_y
    x_width = max_x - min_x

    bin_width = max(x_width, y_width)
    bin_counts = np.zeros((resolution, resolution))

    vectors = vectors.T

    for v in vectors:
        x_bin = get_bin(v[0], min_x, bin_width)
        y_bin = get_bin(v[2], min_y, bin_width)
        bin_counts[y_bin, x_bin] += 1

    return bin_counts


def gaussian_kernel(length=5, sigma=1.0):
    arr = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    gauss = np.exp(-0.5 * (arr**2) / (sigma**2))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def forward_model(incident_ray, normals, distance, origins=None, resolution=256):
    if origins is None:
        origins = np.array([0, 0, 1])  # rg - must be normed

    qs = snell_law(incident_ray, normals)
    qs = c_mat(distance, qs)
    f = qs + origins

    return bin_grid(f, resolution)


def show_pixels(bins):
    plt.imshow(bins, cmap="gray")
    plt.show()


def test_case_forward_model():
    k = normalize(np.array([1, 1, 1]))
    n = normalize(np.array([[7.2, 5.21, 1], [14, 1.5, 1], [11, 0.7, 1], [1, 61, 1]]))

    print(forward_model(k, n, np.array([[1, 2, 3, 4]]).T))


def test_case_bin_counts():
    vectors = np.array([[0.2, 0.8], [0.6, 0.6], [0, 1], [0, 0], [1, 1], [1, 0]])
    # should give :
    # 2 2
    # 1 1
    print(bin_grid(vectors, 2))


def load_data(config):
    normals = np.load(f"../../mitsuba/caustics/outputs/{config}/final_normals.npy")
    positions = np.load(f"../../mitsuba/caustics/outputs/{config}/final_positions.npy")
    normals = normalize(normals)
    return normals, positions


def finalize_validation(incident_rays, normals, positions, resolution, gauss_kernel=True):
    distance = __WALL_DISTANCE - positions.T[1]
    vectors = forward_model(incident_rays, normals, distance, origins=positions, resolution=resolution)
    length = 9
    sigma = 160
    mid = (length - 1) / 2
    if gauss_kernel:
        gauss_stencil = Stencil((resolution, 1), gaussian_kernel(length, sigma), (mid, mid))
        vectors = gauss_stencil(vectors)
    show_pixels(vectors)


def generic_validation(config, incident_rays, resolution, postprocess_incident_rays=None):
    normals, positions = load_data(config)
    if postprocess_incident_rays is None:
        incident_rays = normalize(incident_rays)
    else:
        incident_rays = postprocess_incident_rays(incident_rays, normals, positions, resolution)
    finalize_validation(incident_rays, normals, positions, resolution)


def validation_horizontal_wave():
    generic_validation("wave", np.array([0, 1, 0]), 1024)


def validation_non_horizontal_wave(alpha_x=60, alpha_z=-50):
    generic_validation("wave", np.array([np.sin(np.deg2rad(alpha_x)), 1, np.sin(np.deg2rad(alpha_z))]), 256)


def validation_non_parallel(std_dev=0.005):
    def postprocess_non_parallel(_, normals, positions, resolution):
        nbr = len(normals)
        return normalize(np.array([0, 1, 0]) + std_dev * np.random.randn(nbr, 3))

    generic_validation("wave", None, 256, postprocess_non_parallel)


if __name__ == "__main__":
    validation_horizontal_wave()
