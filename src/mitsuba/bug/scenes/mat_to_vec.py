import numpy as np


def mat_to_vec(mat):
    mat = mat.T
    translation = mat[-1]
    scale_x = np.linalg.norm(mat[0])
    scale_y = np.linalg.norm(mat[1])
    scale_z = np.linalg.norm(mat[2])
    rotation_mat = np.array([mat[0][:-1] / scale_x, mat[1][:-1] / scale_y, mat[2][:-1] / scale_z]).T
    return translation[:-1], [scale_x, scale_y, scale_z], rotation_mat


def str_to_mat(values: str):
    return np.array(list(map(float, values.split()))).reshape((4, 4))


in_str = "-1 0 0 0 0 1 0 1 0 0 -1 6.8 0 0 0 1"
print(str_to_mat(in_str))
print("##########################################################")
x = str_to_mat(in_str)

translation, scale, rotation = mat_to_vec(x)
print(f"Translate: {translation}")
print(f"Scale: {scale}")
print(f"Rotation:\n {rotation}")
