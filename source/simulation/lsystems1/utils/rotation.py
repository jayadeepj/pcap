import numpy as np
import math


def matrix_to_rpy(R, solution=1):
    """Convert a 3x3 transform matrix to roll-pitch-yaw coordinates. Function from urdfpy

    The roll-pitchRyaw axes in a typical URDF are defined as a
    rotation of ``r`` radians around the x-axis followed by a rotation of
    ``p`` radians around the y-axis followed by a rotation of ``y`` radians
    around the z-axis. These are the Z1-Y2-X3 Tait-Bryan angles. See
    Wikipedia_ for more information.

    There are typically two possible roll-pitch-yaw coordinates that could have
    created a given rotation matrix. Specify ``solution=1`` for the first one
    and ``solution=2`` for the second one.

    Parameters
    ----------
    R : (3,3) float
        A 3x3 homogenous rotation matrix.
    solution : int
        Either 1 or 2, indicating which solution to return.

    Returns
    -------
    coords : (3,) float
        The roll-pitch-yaw coordinates in order (x-rot, y-rot, z-rot).
    """
    R = np.asanyarray(R)
    r = 0.0
    p = 0.0
    y = 0.0

    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2, 0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0, 1], R[0, 2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        if solution == 1:
            p = -np.arcsin(R[2, 0])
        else:
            p = np.pi + np.arcsin(R[2, 0])
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))
    return np.array([r, p, y])


def get_rotation_matrix(vec1, vec2):
    """ Get the rotation matrix to align vector vec1 to vector vec2
    Some of the values can end up as nans if you
    """
    vec2 = vec2 / np.linalg.norm(vec2)  # Normalize vec2
    vec1 = vec1 / np.linalg.norm(vec1)  # Normalize vec1

    v = np.cross(vec1, vec2)  # axis
    c = np.dot(vec1, vec2)  # angle

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                     [v3, 0, -v1],
                     [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def calculate_rpy(l1_start, l1_end, l2_start, l2_end):
    """ Given 2 lines l1 and l2 represented by their corresponding start and end points,
      compute the roll, pitch and yaw radians required to turn l1 in the direction of l2. """

    x1, y1, z1 = l1_start.x, l1_start.y, l1_start.z
    x2, y2, z2 = l1_end.x, l1_end.y, l1_end.z

    x3, y3, z3 = l2_start.x, l2_start.y, l2_start.z
    x4, y4, z4 = l2_end.x, l2_end.y, l2_end.z

    dir_vector1 = np.array([x2 - x1, y2 - y1, z2 - z1])
    dir_vector2 = np.array([x4 - x3, y4 - y3, z4 - z3])

    # Handle case where both direction vectors are almost zero
    epsilon = 1e-6
    if np.linalg.norm(dir_vector1) < epsilon or np.linalg.norm(dir_vector2) < epsilon:
        return 0, 0, 0

    # Normalize direction vectors
    dir_vector1_normalized = dir_vector1 / np.linalg.norm(dir_vector1)
    dir_vector2_normalized = dir_vector2 / np.linalg.norm(dir_vector2)

    r_mat = get_rotation_matrix(dir_vector2_normalized, dir_vector1_normalized)

    rpy = [i for i in matrix_to_rpy(r_mat)]

    assert math.isnan(rpy[0]) | math.isnan(rpy[1]) | math.isnan(rpy[2]) is False
    return rpy[0], rpy[1], rpy[2]

