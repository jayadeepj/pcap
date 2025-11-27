""" Domain objects for txn from/to the real robot. functions in this file are invoked during sim2real.
 Can handle transfer of both joint space and cartesian space values.
 """

import torch

_kinova_arm_link_names = [f"j2n6s300_link_{i + 1}" for i in range(6)]
_kinova_finger_link_names = [f"j2n6s300_link_finger_{i + 1}" for i in range(3)]

# these are only used to create pc, actual kinova would include additional links.
kinova_pc_link_names = [*_kinova_arm_link_names, *_kinova_finger_link_names]
kinova_pc_link_cnt = len(kinova_pc_link_names)


class KinovaMetricsR:
    """ Read-only measured 'Raw' metrics  from kinova ros topics. """

    def __init__(self, robot_dof_pos, robot_dof_vel, robot_dof_torque,
                 hand_pos, hand_rot, robot_link_cart_pos, robot_link_cart_rot,

                 transform_to_sim=False):
        self.robot_dof_pos = robot_dof_pos  # 6 dof positions, list of size 6
        self.robot_dof_vel = robot_dof_vel  # 6 dof velocities,  list of size 6
        self.robot_dof_torque = robot_dof_torque  # 6 dof joint torques from Kinova arm , list of size 6
        # j2n6s300_end_effector  p , list of size 3 &  r , list of size 4

        if transform_to_sim:
            self.hand_pos = self._trans_real_to_sim_pos(r_pos=hand_pos)
            self.hand_rot = self._trans_real_to_sim_rot(r_rot=hand_rot)

            # list n x 3; n = number of links in kinova_link_names
            self.robot_link_cart_pos = [self._trans_real_to_sim_pos(r_pos=pos) for pos in robot_link_cart_pos]
            # list n x 4; n = number of links in kinova_link_names
            self.robot_link_cart_rot = [self._trans_real_to_sim_rot(r_rot=rot) for rot in robot_link_cart_rot]

        else:
            self.hand_pos = hand_pos
            self.hand_rot = hand_rot

            self.robot_link_cart_pos = robot_link_cart_pos
            self.robot_link_cart_rot = robot_link_cart_rot

    @staticmethod
    def _trans_real_to_sim_pos(r_pos):
        """ This transformation is required because of the mismatch between real and sim coordinates."""
        real_x, real_y, real_z = r_pos

        sim_x = 1 + real_x * -1
        sim_y = real_y * -1
        sim_z = real_z

        # These values are verified to be correct
        # This works except for an occasional sign flip but A quaternion q = [x, y, z, w]
        # and its negation -q = [-x, -y, -z, -w] represent the same orientation.
        # do an additional normalisation in RL for both sim and real
        return [sim_x, sim_y, sim_z]

    @staticmethod
    def trans_real_to_sim_pos_t(r_pos_t):
        """
        Transforms real-world coordinates to simulation coordinates.

        Args:
        r_pos_t (torch.Tensor): A tensor of size nx3 representing point cloud coordinates.

        Returns:
        torch.Tensor: A tensor of size nx3 representing transformed point cloud coordinates.
        """
        sim_x = 1 - r_pos_t[:, 0]
        sim_y = -r_pos_t[:, 1]
        sim_z = r_pos_t[:, 2]

        return torch.stack([sim_x, sim_y, sim_z], dim=1)

    @staticmethod
    def _trans_real_to_sim_rot(r_rot):
        """ This transformation is required because of the mismatch between real and sim coordinates."""

        real_qx, real_qy, real_qz, real_qw = r_rot

        sim_qx = real_qy
        sim_qy = -1 * real_qx
        sim_qz = -1 * real_qw
        sim_qw = real_qz

        # TODO: verify if the below coordinates are right.
        return [sim_qx, sim_qy, sim_qz, sim_qw]

    @staticmethod
    def trans_real_to_sim_rot_t(r_rot_t):
        """
        Transforms real-world quaternion rotations to simulation quaternion rotations.

        Args:
        r_rot_t (torch.Tensor): A tensor of size nx4 representing quaternions in real-world coordinates.

        Returns:
        torch.Tensor: A tensor of size nx4 representing transformed quaternions in simulation coordinates.
        """
        real_qx = r_rot_t[:, 0]
        real_qy = r_rot_t[:, 1]
        real_qz = r_rot_t[:, 2]
        real_qw = r_rot_t[:, 3]

        sim_qx = real_qy
        sim_qy = -real_qx
        sim_qz = -real_qw
        sim_qw = real_qz

        return torch.stack([sim_qx, sim_qy, sim_qz, sim_qw], dim=1)

    # sim to real tranformation
    @staticmethod
    def trans_sim_to_real_pos_t(sim_pos_t):
        """
        Transforms simulation coordinates back to real-world coordinates.

        Args:
        sim_pos_t (torch.Tensor): A tensor of size nx3 representing positions in simulation coordinates.

        Returns:
        torch.Tensor: A tensor of size nx3 representing transformed positions in real-world coordinates.
        """
        real_x = 1 - sim_pos_t[:, 0]
        real_y = -sim_pos_t[:, 1]
        real_z = sim_pos_t[:, 2]

        return torch.stack([real_x, real_y, real_z], dim=1)

    @staticmethod
    def trans_sim_to_real_rot_t(sim_rot_t):
        """
        Transforms simulation quaternion rotations back to real-world quaternions.

        Args:
        sim_rot_t (torch.Tensor): A tensor of size nx4 representing quaternions in simulation coordinates.

        Returns:
        torch.Tensor: A tensor of size nx4 representing quaternions in real-world coordinates.
        """
        sim_qx = sim_rot_t[:, 0]
        sim_qy = sim_rot_t[:, 1]
        sim_qz = sim_rot_t[:, 2]
        sim_qw = sim_rot_t[:, 3]

        real_qx = -sim_qy
        real_qy = sim_qx
        real_qz = sim_qw
        real_qw = -sim_qz

        return torch.stack([real_qx, real_qy, real_qz, real_qw], dim=1)


def validate_km_tensor(km_instance):
    """Validate dimensions of tensors."""
    assert km_instance.robot_dof_pos_t.shape == torch.Size([1, 6]), "robot_dof_pos should be a 1x6 tensor"
    assert km_instance.robot_dof_vel_t.shape == torch.Size([1, 6]), "robot_dof_vel should be a 1x6 tensor"
    assert km_instance.robot_dof_torque_t.shape == torch.Size([1, 6]), "robot_dof_torque should be a 1x6 tensor"
    assert km_instance.hand_pos_t.shape == torch.Size([1, 3]), "hand_pos should be a 1x3 tensor"
    assert km_instance.hand_rot_t.shape == torch.Size([1, 4]), "hand_rot should be a 1x4 tensor"

    assert km_instance.robot_link_cart_pos_t.shape == torch.Size([1, kinova_pc_link_cnt, 3]), "should be a 1xnx3"
    assert km_instance.robot_link_cart_rot_t.shape == torch.Size([1, kinova_pc_link_cnt, 4]), "should be a 1xnx4"


class KinovaMetricsT:
    """ Measured metrics 'Tensor' from kinova ros topics. """

    def __init__(self):
        self.robot_dof_pos_t = None  # 6d tensor
        self.robot_dof_vel_t = None  # 6d tensor
        self.robot_dof_torque_t = None  # 6d tensor
        self.hand_pos_t = None  # j2n6s300_end_effector  p , 3d tensor
        self.hand_rot_t = None  # j2n6s300_end_effector  r , 4d tensor

        self.robot_link_cart_pos_t = None  # nx3d tensor
        self.robot_link_cart_rot_t = None  # nx4d tensor

    @classmethod
    def to_tensor(cls, kinova_metrics_r, device):
        kinova_metrics_t = cls()

        if kinova_metrics_r.robot_dof_pos is not None:
            kinova_metrics_t.robot_dof_pos_t = torch.tensor(kinova_metrics_r.robot_dof_pos, device=device).unsqueeze(0)

        if kinova_metrics_r.robot_dof_vel is not None:
            kinova_metrics_t.robot_dof_vel_t = torch.tensor(kinova_metrics_r.robot_dof_vel, device=device).unsqueeze(0)

        if kinova_metrics_r.robot_dof_torque is not None:
            kinova_metrics_t.robot_dof_torque_t = torch.tensor(kinova_metrics_r.robot_dof_torque,
                                                               device=device).unsqueeze(0)

        if kinova_metrics_r.hand_pos is not None:
            kinova_metrics_t.hand_pos_t = torch.tensor(kinova_metrics_r.hand_pos, device=device).unsqueeze(0)

        if kinova_metrics_r.hand_rot is not None:
            kinova_metrics_t.hand_rot_t = torch.tensor(kinova_metrics_r.hand_rot, device=device).unsqueeze(0)

        if kinova_metrics_r.robot_link_cart_pos is not None:
            kinova_metrics_t.robot_link_cart_pos_t = torch.tensor(kinova_metrics_r.robot_link_cart_pos,
                                                                  device=device).unsqueeze(0)

        if kinova_metrics_r.robot_link_cart_rot is not None:
            kinova_metrics_t.robot_link_cart_rot_t = torch.tensor(kinova_metrics_r.robot_link_cart_rot,
                                                                  device=device).unsqueeze(0)

        validate_km_tensor(kinova_metrics_t)  # validate dimensions before responding

        return kinova_metrics_t


if __name__ == '__main__':
    # Example usage:
    _real_pos = [0.21428863501548767, -0.5376197695732117, 0.5049141645431519]
    _sim_pos = KinovaMetricsR._trans_real_to_sim_pos(_real_pos)
    print("Pos in simulation: ", _sim_pos)

    _real_rot = [0.687, 0.213, 0.333, 0.610]
    _sim_rot = KinovaMetricsR._trans_real_to_sim_rot(_real_rot)
    print("Rot in simulation: ", _sim_rot)

    _real_pos = [0.5149, 0.0555, 0.8501]
    _sim_pos = KinovaMetricsR._trans_real_to_sim_pos(_real_pos)
    print("Pos in simulation: ", _sim_pos)

    _real_pos = torch.tensor([[0.21428863501548767, -0.5376197695732117, 0.5049141645431519]])
    _sim_pos = KinovaMetricsR.trans_real_to_sim_pos_t(_real_pos)
    print("Pos in simulation: ", _sim_pos)

    # Pos in simulation: [0.7857113649845123, 0.5376197695732117, 0.5049141645431519]
    # Rot in simulation: [0.213, -0.687, -0.61, 0.333]
    # Pos in simulation: [0.4851, -0.0555, 0.8501]
