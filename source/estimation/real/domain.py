import math
import torch
from enum import Enum

# valid_dims = [0, 0, 1, 0, 0, 0] means set all axis except z to zero
# By setting specific dim to zero, the motion/force in those dimensions can be removed
valid_dims = [0, 0, 1, 0, 0, 0]


class ArmType(str, Enum):
    kinova = 'kinova'
    franka = 'franka'


class MeasurementConstants:
    """ Preset Constants at the time of measurement using physical arm"""

    def __init__(self, arm_type=ArmType.kinova):
        # time step len for the physical arm. Should be == the publish rate of the topics

        if arm_type == ArmType.kinova:
            self.arm_dt = 1.0 / 10.0

        if arm_type == ArmType.franka:
            self.arm_dt = 1.0 / 10.0

        # frequency at which to retrieve ros messages, best to match topic publish freq
        self.measure_freq = int(1 / self.arm_dt)

        # num of seconds to measure forces/pos/velocity before applying any external force
        self.zero_state_duration_sec = 2
        # num of time steps to measure forces/pos/velocity before applying any external force
        self.zero_state_offset = self.zero_state_duration_sec * self.measure_freq


class Pose:

    def __init__(self, x, y, z, ox, oy, oz):
        self.x = x
        self.y = y
        self.z = z

        self.ox = ox
        self.oy = oy
        self.oz = oz
        self.ignore_dims()

    def ignore_dims(self):
        """ By setting specific dim to zero, the motion/force in those dims can be removed """
        for ax, validity in zip([self.x, self.y, self.z, self.ox, self.oy, self.oz], valid_dims):
            ax = ax * validity

    @classmethod
    def from_kinova_msg_str(cls, sensor_msg):
        def extract(_msg_strs, idx):
            temp = _msg_strs[idx].split(": ")
            return float(temp[1])

        _msg_strs = str(sensor_msg).split("\n")

        x = extract(_msg_strs, 0)
        y = extract(_msg_strs, 1)
        z = extract(_msg_strs, 2)

        ox = extract(_msg_strs, 3)
        oy = extract(_msg_strs, 4)
        oz = extract(_msg_strs, 5)
        return cls(x, y, z, ox, oy, oz)

    def position_vec(self):
        return [self.x, self.y, self.z]

    def orientation_vec(self):
        return [self.ox, self.oy, self.oz]

    def pose_vec(self):
        return [self.x, self.y, self.z, self.ox, self.oy, self.oz]

    def pose(self):
        return self._tensor(self.pose_vec())

    def position(self):
        return self._tensor(self.position_vec())

    def orientation(self):
        return self._tensor(self.orientation_vec())

    @staticmethod
    def _tensor(vec):
        return torch.FloatTensor(vec).unsqueeze(0)

    @staticmethod
    def home_pose():
        home = [0.212, -0.257, 0.509, 1.637, 1.113, 0.134]
        return Pose._tensor(home)

    def __str__(self):
        return f'Pose(\'{self.x}\', {self.y}, {self.z}, {self.ox}, {self.oy}, {self.oz})'


class Wrench:

    def __init__(self, fx, fy, fz, tx, ty, tz):
        self.fx = fx
        self.fy = fy
        self.fz = fz

        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.force_mag = get_vector_magnitude([fx, fy, fz])
        self.torque_mag = get_vector_magnitude([tx, ty, tz])

        self.ignore_dims()

    def ignore_dims(self):
        """ By setting specific dim to zero, the motion/force in those dimensions can be removed """
        for ax, validity in zip([self.fx, self.fy, self.fz, self.tx, self.ty, self.tz], valid_dims):
            ax = ax * validity

    @classmethod
    def from_kinova_msg_str(cls, sensor_msg):
        fx = sensor_msg.wrench.force.x
        fy = sensor_msg.wrench.force.y
        fz = sensor_msg.wrench.force.z

        tx = sensor_msg.wrench.torque.x
        ty = sensor_msg.wrench.torque.y
        tz = sensor_msg.wrench.torque.z
        return cls(fx, fy, fz, tx, ty, tz)

    def force_vec(self):
        return [self.fx, self.fy, self.fz]

    def torque_vec(self):
        return [self.tx, self.ty, self.tz]

    @staticmethod
    def _tensor(vec):
        return torch.FloatTensor(vec).unsqueeze(0)

    def force(self):
        return self._tensor(self.force_vec())

    @staticmethod
    def home_force():
        home = [0., 0., 0., 0., 0., 0.]
        return Wrench._tensor(home[0:3])

    def __str__(self):
        return f'Wrench(\'{self.fx}\', {self.fy}, {self.fz}, {self.tx}, {self.ty}, {self.tz})'


def get_vector_magnitude(vector):
    return math.sqrt(pow(vector[0], 2) + pow(vector[1], 2) + pow(vector[2], 2))
