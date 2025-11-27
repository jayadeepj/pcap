import enum
import math
from collections import namedtuple
import simulation.basic.conf.yaml_parser as parser


class Shapes(enum.Enum):
    Cylinder = 1
    Cube = 2


def volume(height, radius, shape):
    if shape == Shapes.Cylinder:
        return math.pi * (radius ** 2) * height

    raise ValueError("Not Implemented")


def mass(height, radius, shape):
    """ mass = density * vol """
    density = parser.config()['physics']['density']
    vol = volume(height, radius, shape)
    return round(density * vol, 6)


# calculate moment of inertia of a cylinder
def moment_of_inertia_cylinder(height, radius, shape):
    m = mass(height, radius, shape)
    xx = round(m * (3 * (radius ** 2) + (height ** 2)) / 12, 3)
    yy = xx
    zz = round(m * (radius ** 2) / 2, 3)
    xy, xz, yz = 0, 0, 0
    moi = namedtuple('moi', ['xx', 'yy', 'zz', 'xy', 'xz', 'yz'])
    return moi(xx=xx, yy=yy, zz=zz, xy=xy, xz=xz, yz=yz)


def motor_force(height, radius, shape):
    """ F = m * acceleration. This is only an indicative force to be applied
    This has to decay exponentially due to the weight of all branches above it
    Average human push force in standing position is 100-200 Newtons."""

    acceleration = parser.config()['physics']['acceleration']
    m = mass(height, radius, shape)
    friction, damping = friction_n_damping(m)
    return m * acceleration + friction + damping


def friction_n_damping(m):
    friction_factor = parser.config()['physics']['friction_factor']
    damping_factor = parser.config()['physics']['damping_factor']
    friction = m * friction_factor
    damping = m * damping_factor
    fd_pair = namedtuple('fd_pair', ['friction', 'damping'])
    return fd_pair(friction=round(friction, 3), damping=round(damping, 3))


def joint_supported_mass(branch_child_cnts, branch_masses):
    """ Provides the supported masses at each joint based on the cumulative sum of masses of all children
    E.g. branch_child_cnts = [2, 2, 0], branch_masses = [100, 40, 17] => supported_mass = [248, 74, 17]
    """
    supported_mass = [0 for _ in range(len(branch_child_cnts))]
    supported_mass[-1] = branch_masses[-1]

    for i in reversed(range(len(branch_child_cnts) - 1)):
        supported_mass[i] = branch_masses[i] + branch_child_cnts[i] * supported_mass[i + 1]

    return supported_mass


def joint_support_mass_ratios(supported_mass):
    """ Given the supported mass vector create a vector of cumulative mass proportion
     E.g. supported_mass =[50, 30, 10, 5]  proportional_force = [0, 1, 2, 1]
     The first index is 0, because there is no joint at trunk is fixed.
     """

    if len(supported_mass) < 2:
        raise ValueError("Invalid length for supported masses")

    support_mass_ratios = [0 for _ in range(len(supported_mass))]
    support_mass_ratios[0] = 0
    support_mass_ratios[-1] = 1

    for i in range(0, len(supported_mass)):
        support_mass_ratios[i] = supported_mass[i] / supported_mass[-1]

    return [round(f, 3) for f in support_mass_ratios]


print(joint_support_mass_ratios([50, 30, 10, 5]))


def test():
    print(friction_n_damping(m=20))
    moment_of_inertia_cylinder(height=1.066, radius=0.1, shape=Shapes.Cylinder)
