import enum
import math
from collections import namedtuple


class Shapes(enum.Enum):
    Cylinder = 1
    Cube = 2


def volume(height, radius, shape):
    if shape == Shapes.Cylinder:
        return math.pi * (radius ** 2) * height

    raise ValueError("Not Implemented")


def mass(height, radius, density, shape=Shapes.Cylinder):
    """ mass = density * vol """

    vol = volume(height, radius, shape)
    return round(density * vol, 6)


# calculate moment of inertia of a cylinder
def moment_of_inertia_cylinder(height, radius, density, shape=Shapes.Cylinder):
    m = mass(height, radius, density, shape)
    xx = round(m * (3 * (radius ** 2) + (height ** 2)) / 12, 3)
    yy = xx
    zz = round(m * (radius ** 2) / 2, 3)
    xy, xz, yz = 0, 0, 0
    moi = namedtuple('moi', ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])
    return moi(xx=xx, yy=yy, zz=zz, xy=xy, xz=xz, yz=yz)
