""" Use isaac gym line api to draw shapes for locating the position for debug"""

import numpy as np


def draw_cube(center=np.array([0., 0., 0.]), side_length=0.10):
    """ Given a centre (x,y,z), draw a cube with the given side length."""
    half_side = side_length / 2.0

    center[2] = center[2] + half_side if center[2] == 0 else center[2]  # lift a bit in case the z=0

    # Define the vertices of the cube
    vertices = np.array([
        [center[0] - half_side, center[1] - half_side, center[2] - half_side],
        [center[0] + half_side, center[1] - half_side, center[2] - half_side],
        [center[0] + half_side, center[1] + half_side, center[2] - half_side],
        [center[0] - half_side, center[1] + half_side, center[2] - half_side],
        [center[0] - half_side, center[1] - half_side, center[2] + half_side],
        [center[0] + half_side, center[1] - half_side, center[2] + half_side],
        [center[0] + half_side, center[1] + half_side, center[2] + half_side],
        [center[0] - half_side, center[1] + half_side, center[2] + half_side]
    ])

    # Define the cube's edges by connecting vertices
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting edges
    ]

    # Create line vertices for the cube edges
    cube_lines = np.array([
        [vertices[edge[0]], vertices[edge[1]]] for edge in edges
    ])

    return cube_lines


def draw_square(center=np.array([0., 0., 0.]), side_length=0.10):
    """ Given a centre (x,y,z), draw a square ignoring the z-axis"""

    half_side = side_length / 2.0

    # Define the vertices of the square
    vertices = np.array([
        [center[0] - half_side, center[1] + half_side, center[2]],
        [center[0] + half_side, center[1] + half_side, center[2]],
        [center[0] + half_side, center[1] - half_side, center[2]],
        [center[0] - half_side, center[1] - half_side, center[2]],
        [center[0] - half_side, center[1] + half_side, center[2]]  # Closing the square
    ])

    # Add the square as lines
    line_vertices_square = np.array([
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]]
    ])
    diagonal_vertices = np.array([
        [vertices[0], vertices[2]],
        [vertices[1], vertices[3]]
    ])

    # Combine square and diagonal vertices
    lines = np.vstack([line_vertices_square, diagonal_vertices])

    return lines
