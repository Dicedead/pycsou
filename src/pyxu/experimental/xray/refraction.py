import numpy as np

import pyxu.info.ptype as pxt
import pyxu.util.array_module as pxa


def normalize(k: pxt.NDArray) -> pxt.NDArray:
    """
    Normalize an array of vectors.

    Parameters
    ----------
    k: (N,D)

    Returns
    -------
    Normalized (N,D) version of k, along each D dimensional vector line.

    """
    return k / np.linalg.norm(k, axis=-1)[:, np.newaxis]


def zero_z(k: pxt.NDArray) -> pxt.NDArray:
    """
    Zero out last array dimension

    Parameters
    ----------
    k: (N,D)

    Returns
    -------
    k with a 0 component in its last dimension

    """
    q = k.copy()
    q[:, -1] = 0
    return q


def snell_vectorial(n1: pxt.Real, n2: pxt.Real, k_incident: pxt.NDArray, n_normal: pxt.NDArray) -> pxt.NDArray:
    """
    Compute refracted ray.

    Note:
    If n1 > n2 total reflectance may happen for sufficiently oblique angles.
    The resulting refracted ray is a nan vector, since that ray effectively does not exist.

    Parameters
    ----------
    n1: refractive index of incoming ray's medium
    n2: refractive index of refracted ray's medium
    k_incident: normalized incident ray (N, D)
    n_normal: normalized normal ray (N, D)

    Returns
    -------
    Refracted rays for each (k_incident, n_normal) pair, (N, D)

    """
    xp = pxa.get_array_module(k_incident)
    ratio = n1 / n2
    first_term = ratio * xp.cross(n_normal, xp.cross(-n_normal, k_incident, axis=-1), axis=-1)
    inner = 1 - (first_term**2).sum(axis=-1)
    mask = inner < 0
    inner[mask] = 0.0
    second_term = (-n_normal) * xp.sqrt(inner)[:, xp.newaxis]
    temp = first_term + second_term
    temp[mask] = xp.nan
    return temp


def intersection_line_sphere(
    radius: pxt.Real, height: pxt.NDArray, t_init: pxt.NDArray, n_dir: pxt.NDArray
) -> pxt.NDArray:
    """
    Compute (if any) the position of an intersection of a ray with a cylinder.

    Parameters
    ----------
    radius: (1,) cylinder radius
    height: (2,) min and max cylinder heights
    t_init: (N,D) position of ray outside cylinder
    n_dir: (N,D) normalized direction of ray outside cylinder

    Returns
    -------
    Position of intersection with the sphere, (N,D)
    If no intersection: returns the nan vector.

    """
    xp = pxa.get_array_module(t_init)

    t_init_2 = t_init[:, :2]
    n_dir_2 = n_dir[:, :2]
    dot_prod = xp.einsum("ij,ij->i", t_init_2, n_dir_2)

    radicand = (dot_prod**2) + radius * radius - (t_init_2**2).sum(axis=-1)

    no_intersec_mask = radicand < 0 | xp.any(xp.isnan(n_dir), axis=-1)
    radicand[no_intersec_mask] = 0.0

    sqrt_rad = xp.sqrt(radicand)
    x1 = -dot_prod + sqrt_rad
    x2 = -dot_prod - sqrt_rad

    v1 = t_init + n_dir * x1[:, xp.newaxis]
    v2 = t_init + n_dir * x2[:, xp.newaxis]

    t_out = v1
    v2_mask = xp.linalg.norm(v1 - t_init, axis=-1) > xp.linalg.norm(v2 - t_init, axis=-1)
    tmp = v2[v2_mask]
    t_out[v2_mask] = tmp

    t_out[no_intersec_mask] = xp.nan
    t_out[(t_out[:, -1] > height[1]) | (t_out[:, -1] < height[0])] = xp.nan
    return t_out


def intersection_polymer(c_spec: pxt.NDArray, t_in: pxt.NDArray, n_refracted: pxt.NDArray) -> pxt.NDArray:
    """
    Compute (if any) the position of an intersection of a ray with the polymer.

    Parameters
    ----------
    c_spec: (2,) glass thickness and cylinder external diameter
    t_in: (N,D) position on cylinder border
    n_refracted: (N,D) normalized direction inside cylinder (as in inside the glass)

    Returns
    -------
    Position of intersection with the polymer, (N,D)
    If no intersection: returns the nan vector.

    """
    inner_radius = c_spec[1] / 2 - c_spec[0]
    return intersection_line_sphere(inner_radius, c_spec[-2:], t_in, n_refracted)


def intersection_cylinder(c_spec: pxt.NDArray, t_init: pxt.NDArray, n_in: pxt.NDArray) -> pxt.NDArray:
    """
    Compute (if any) the position of an intersection of a ray with the polymer.

    Parameters
    ----------
    c_spec: (2,) glass thickness and cylinder external diameter.
    t_init: (N,3) initial position outside cylinder.
    n_in: (N,3) normalized direction outside cylinder

    Returns
    -------
    Position of intersection with the cylinder, (N,3)
    If no intersection: returns the nan vector.

    """
    outer_radius = c_spec[1] / 2
    return intersection_line_sphere(outer_radius, c_spec[-4:-2], t_init, n_in)


def refract(
    n_in: pxt.NDArray, t_init: pxt.NDArray, r_spec: pxt.NDArray, c_spec: pxt.NDArray
) -> (pxt.NDArray, pxt.NDArray):
    """

    Parameters
    ----------
    n_in: (N,D) ray direction outside cylinder (not necessarily normalized)
    t_init: (N,D) ray position outside cylinder
    r_spec: (3,) index of refraction of air/cylinder/jelly
    c_spec: (6,) glass thickness, cylinder external diameter, glass min height, glass max height, polymer min height,
                polymer max height

    Returns
    -------
    (n_out, t_out): ray direction/position inside cylinder, ((N,D), (N,D))
    Both vectors are nan-full if incoming ray does not enter jelly

    """
    n_in = normalize(n_in)

    xp = pxa.get_array_module(n_in)
    n_air, n_cyl, n_gel = r_spec[0], r_spec[1], r_spec[2]
    t_in = intersection_cylinder(c_spec, t_init, n_in)
    n_refracted = snell_vectorial(n_air, n_cyl, n_in, normalize(zero_z(t_in)))
    t_out = intersection_polymer(c_spec, t_in, n_refracted)
    n_out = snell_vectorial(n_cyl, n_gel, n_refracted, normalize(zero_z(t_out)))

    nan_mask = xp.any(xp.isnan(t_out), axis=-1)
    n_out[nan_mask] = xp.nan
    return n_out, t_out


def sine(v, w):
    sines = np.zeros(len(v))
    for i in range(len(v)):
        sines[i] = np.sqrt(1 - (v[i] @ w[i]) ** 2)

    return sines


def test_snell():
    k = np.array([[1, -1, 0], [-1, -4, -5], [-1, -2, 6], [5, 3, 2]])
    k = k / np.linalg.norm(k, axis=-1)[:, np.newaxis]
    k = k
    n = np.array([[0, 1, 0], [1, 5, 1], [1, 2, 9], [5, 1, 4]])
    n = n / np.linalg.norm(n, axis=-1)[:, np.newaxis]
    n1 = 1.9
    n2 = 1
    res1 = snell_vectorial(n1, n2, k, n)

    print(snell_vectorial(n1, n2, k, n))
    print(k)
    print(np.linalg.norm(res1, axis=-1))
    print(np.linalg.norm(k, axis=-1))
    print(n2 * sine(res1, n) - n1 * sine(k, n))


def test_f():
    c_spec = [1, 8, 0, 10, 0, 10]
    r_spec = [1, 1.4, 1.45]
    n_in = np.array([[-1, 0, 0], [-1, 0.4, 0.1]])

    t_init = np.array([[20, 0, 5], [-10, 0, 5]])

    n_out, t_out = refract(n_in, t_init, r_spec, c_spec)

    print(n_out)
    print(t_out)


if __name__ == "__main__":
    test_f()
