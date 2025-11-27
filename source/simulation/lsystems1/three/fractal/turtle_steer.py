""" Functions to turn, rotate and spin the turtle
partly taken/inspired from from https://github.com/ThomasLENNE/L-system
HLU Description from Prusinkiewicz, P., & Lindenmayer, A. (2012). The algorithmic beauty of plants. . Page 18, Sec 1.5.
"""

import math
import numpy as np


def rotate_around_u(angle, hlu):
    """
    Rotate the turtle around U by the given angle.

    Parameters:
    - angle (float): Angle in radians.
    - hlu (tuple): Turtle orientation as a tuple of lists.

    Returns:
    - Updated turtle orientation (tuple of lists).
    """
    hlu = np.array(hlu).T

    # The rotation_matrix is from abop page 19
    rotation_matrix = np.array([
        [math.cos(angle), math.sin(angle), 0],
        [-math.sin(angle), math.cos(angle), 0],
        [0, 0, 1]
    ])
    hlu = np.dot(hlu, rotation_matrix)
    H, L, U = hlu.T[0], hlu.T[1], hlu.T[2]
    return H.tolist(), L.tolist(), U.tolist()


def rotate_around_h(angle, hlu):
    """
    Rotate the turtle around H by the given angle.

    Parameters:
    - angle (float): Angle in radians.
    - hlu (tuple): Turtle orientation as a tuple of lists.

    Returns:
    - Updated turtle orientation (tuple of lists).
    """
    hlu = np.array(hlu).T

    # The rotation_matrix is from abop page 19
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, math.cos(angle), -math.sin(angle)],
        [0, math.sin(angle), math.cos(angle)]
    ])
    hlu = np.dot(hlu, rotation_matrix)
    H, L, U = hlu.T[0], hlu.T[1], hlu.T[2]
    return H.tolist(), L.tolist(), U.tolist()


def rotate_around_l(angle, hlu):
    """
    Rotate the turtle around L by the given angle.

    Parameters:
    - angle (float): Angle in radians.
    - hlu (tuple): Turtle orientation as a tuple of lists.

    Returns:
    - Updated turtle orientation (tuple of lists).
    """
    hlu = np.array(hlu).T

    # The rotation_matrix is from abop page 19
    rotation_matrix = np.array([
        [math.cos(angle), 0, -math.sin(angle)],
        [0, 1, 0],
        [math.sin(angle), 0, math.cos(angle)]
    ])
    hlu = np.dot(hlu, rotation_matrix)
    H, L, U = hlu.T[0], hlu.T[1], hlu.T[2]
    return H.tolist(), L.tolist(), U.tolist()


def keep_l_horizontal(hlu):
    """
    Keep L horizontal.

    Parameters:
    - hlu (tuple): Turtle orientation.

    Returns:
    - Updated turtle orientation (tuple of lists).
    """
    H = hlu[0]
    [xh, yh, zh] = H
    V = [0, 0, 1]
    L = [-yh, -xh, 0]
    U = [xh * zh, -zh * yh, -xh ** 2 - yh ** 2]
    return H, L, U


def normalize(vector):
    """
    Normalize a vector.

    Parameters:
    - vector (list): Input vector.

    Returns:
    - Normalized vector (list).
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def torque(hlu, tropism):
    """
    Calculate the torque (H * tropism) for the given turtle orientation.

    Parameters:
    - hlu (tuple): Turtle orientation.
    - tropism (list): Tropism vector.

    Returns:
    - Torque vector (list).
    """
    H = hlu[0]
    [xh, yh, zh] = H
    [xt, yt, zt] = tropism
    torque = [yh * zt - zh * yt, zh * xt - xh * zt, xh * yt - yh * xt]
    return torque


def rotate(hlu, axis, angle):
    """
    Rotate the turtle vector around the given axis by the given angle.

    Parameters:
    - hlu (tuple): Turtle orientation before rotation.
    - axis (list): Axis vector.
    - angle (float): Rotation angle in radians.

    Returns:
    - Updated turtle orientation (tuple of lists).
    """
    axis = normalize(axis)
    [ax, ay, az] = axis
    c = math.cos(angle)
    s = math.sin(angle)
    rotation_matrix = np.array([
        [ax ** 2 * (1 - c) + c, ax * ay * (1 - c) - az * s, ax * az * (1 - c) + ay * s],
        [ax * ay * (1 - c) + az * s, ay ** 2 * (1 - c) + c, ay * az * (1 - c) - ax * s],
        [ax * az * (1 - c) - ay * s, ay * az * (1 - c) + ax * s, az ** 2 * (1 - c) + c]
    ])
    [H, L, U] = hlu
    H = np.array(H)
    L = np.array(L)
    U = np.array(U)
    H = np.dot(rotation_matrix, H)
    L = np.dot(rotation_matrix, L)
    U = np.dot(rotation_matrix, U)
    return H.tolist(), L.tolist(), U.tolist()


def apply_tropism(hlu, tropism):
    """
    Apply tropism to the turtle orientation.

    Parameters:
    - hlu (tuple): Turtle orientation.
    - tropism (tuple): Tropism vector.

    Returns:
    - Updated turtle orientation (tuple of lists).
    """
    M = torque(hlu, tropism)  # Rotation axis
    alpha = np.linalg.norm(M)  # Rotation angle
    H, L, U = rotate(hlu, M, alpha)
    return H, L, U
