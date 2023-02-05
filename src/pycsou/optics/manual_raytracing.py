import matplotlib.pyplot as plt
import numpy as np


def c_mat(c, mat):
    return (mat.T * c).T


def normalize(vs):
    vs = vs.T
    return (vs / np.linalg.norm(vs, axis=0)).T


def snell_law(incident_ray, normal, n1=1.000277, n2=1.5):
    # refraction indices from Mitsuba test

    k = incident_ray  # shorthand
    n = normal  # shorthand

    if np.isclose(n[1], 0).any():
        raise ValueError("Avoiding a division by 0")

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


def generic_validation(config, incident_rays, resolution):
    incident_rays = normalize(incident_rays)

    normals = np.load(f"../../mitsuba/caustics/outputs/{config}/final_normals.npy")
    positions = np.load(f"../../mitsuba/caustics/outputs/{config}/final_positions.npy")

    normals = normalize(normals)

    distance = 7 - positions.T[1]
    vectors = forward_model(incident_rays, normals, distance, origins=positions, resolution=resolution)
    show_pixels(vectors)


def validation_parallel_wave():
    generic_validation("wave", np.array([0, 1, 0]), 256)


def validation_non_parallel_wave(alpha_x=60, alpha_z=-50):
    generic_validation("wave", np.array([np.sin(np.deg2rad(alpha_x)), 1, np.sin(np.deg2rad(alpha_z))]), 256)


if __name__ == "__main__":
    validation_parallel_wave()
