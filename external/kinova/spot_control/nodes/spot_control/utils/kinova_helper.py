import math
from enum import Enum


class Arm:
    def __init__(self, robot_category, robot_category_version, wrist_type, arm_joint_number,
                 robot_mode, finger_number, prefix, finger_max_dist, finger_max_turn):
        self.robot_category = robot_category
        self.robot_category_version = robot_category_version
        self.wrist_type = wrist_type
        self.arm_joint_number = arm_joint_number
        self.robot_mode = robot_mode
        self.finger_number = finger_number
        self.prefix = prefix
        self.finger_max_dist = finger_max_dist
        self.finger_max_turn = finger_max_turn


class CollisionType(Enum):
    """ This enum is to capture trajectories for the collision classifier."""
    FREE = "free"  # for trajectories without any collisions.
    INTERACTION = "interaction"  # for trajectories with collisions.
    NA = "na"  # always use na when the data captured is not used for training collision classifier.


def kinova_robot_type_parser(kinova_robot_type):
    """ Generate the arm details kinova_robot_type """

    robot_category = kinova_robot_type[0]
    robot_category_version = int(kinova_robot_type[1])
    wrist_type = kinova_robot_type[2]
    arm_joint_number = int(kinova_robot_type[3])
    robot_mode = kinova_robot_type[4]
    finger_number = int(kinova_robot_type[5])
    prefix = kinova_robot_type + "_"
    finger_max_dist = 18.9 / 2 / 1000  # max distance for one finger in meter
    finger_max_turn = 6800  # max thread turn for one finger

    return Arm(robot_category, robot_category_version, wrist_type, arm_joint_number,
               robot_mode, finger_number, prefix, finger_max_dist, finger_max_turn)


def euler_xyz_2_quaternion(euler_xyz):
    tx_, ty_, tz_ = euler_xyz[0:3]
    sx = math.sin(0.5 * tx_)
    cx = math.cos(0.5 * tx_)
    sy = math.sin(0.5 * ty_)
    cy = math.cos(0.5 * ty_)
    sz = math.sin(0.5 * tz_)
    cz = math.cos(0.5 * tz_)

    qx_ = sx * cy * cz + cx * sy * sz
    qy_ = -sx * cy * sz + cx * sy * cz
    qz_ = sx * sy * cz + cx * cy * sz
    qw_ = -sx * sy * sz + cx * cy * cz

    Q_ = [qx_, qy_, qz_, qw_]
    return Q_
