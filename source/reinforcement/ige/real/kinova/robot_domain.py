""" Domain objects for txn from/to the real robot. functions in this file are invoked during sim2real """

import torch
from enum import Enum
import pickle


class KinovaMetricsR:
    """ Read-only measured 'Raw' metrics  from kinova ros topics. """

    def __init__(self, robot_dof_pos, robot_dof_vel, robot_dof_torque, hand_pos, hand_rot, transform_to_sim=False):
        self.robot_dof_pos = robot_dof_pos  # 6 dof positions, list of size 6
        self.robot_dof_vel = robot_dof_vel  # 6 dof velocities,  list of size 6
        self.robot_dof_torque = robot_dof_torque  # 6 dof joint torques from Kinova arm , list of size 6
        # j2n6s300_end_effector  p , list of size 3 &  r , list of size 4

        if transform_to_sim:
            self.hand_pos = self._trans_real_to_sim_pos(r_pos=hand_pos)
            self.hand_rot = self._trans_real_to_sim_rot(r_rot=hand_rot)
        else:
            self.hand_pos = hand_pos
            self.hand_rot = hand_rot

    @staticmethod
    def _trans_real_to_sim_pos(r_pos):
        """ This transformation is required because of the mismatch between real and sim coordinates."""
        real_x, real_y, real_z = r_pos

        sim_x = 1 + real_x * -1
        sim_y = real_y * -1
        sim_z = real_z

        # TODO: verify if the below coordinates are right.
        return [sim_x, sim_y, sim_z]

    @staticmethod
    def _trans_real_to_sim_rot(r_rot):
        """ This transformation is required because of the mismatch between real and sim coordinates."""

        real_qx, real_qy, real_qz, real_qw = r_rot

        sim_qx = real_qy
        sim_qy = -1 * real_qx
        sim_qz = -1 * real_qw
        sim_qw = real_qz

        # These values are verified to be correct
        # This works except for an occasional sign flip but A quaternion q = [x, y, z, w]
        # and its negation -q = [-x, -y, -z, -w] represent the same orientation.
        # do an additional normalisation in RL for both sim and real
        return [sim_qx, sim_qy, sim_qz, sim_qw]

    @staticmethod
    def transform_sim_to_real_pos(sim_pos):
        sim_x, sim_y, sim_z = sim_pos

        real_x = 1 - sim_x
        real_y = sim_y * -1
        real_z = sim_z

        return [real_x, real_y, real_z]


def validate_km_tensor(km_instance):
    """Validate dimensions of tensors."""
    assert km_instance.robot_dof_pos_t.shape == torch.Size([1, 6]), "robot_dof_pos should be a 1x6 tensor"
    assert km_instance.robot_dof_vel_t.shape == torch.Size([1, 6]), "robot_dof_vel should be a 1x6 tensor"
    assert km_instance.robot_dof_torque_t.shape == torch.Size([1, 6]), "robot_dof_torque should be a 1x6 tensor"
    assert km_instance.hand_pos_t.shape == torch.Size([1, 3]), "hand_pos should be a 1x3 tensor"
    assert km_instance.hand_rot_t.shape == torch.Size([1, 4]), "hand_rot should be a 1x4 tensor"


class KinovaMetricsT:
    """ Measured metrics 'Tensor' from kinova ros topics. """

    def __init__(self):
        self.robot_dof_pos_t = None  # 6d tensor
        self.robot_dof_vel_t = None  # 6d tensor
        self.robot_dof_torque_t = None  # 6d tensor
        self.hand_pos_t = None  # j2n6s300_end_effector  p , 3d tensor
        self.hand_rot_t = None  # j2n6s300_end_effector  r , 4d tensor

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

        validate_km_tensor(kinova_metrics_t)  # validate dimensions before responding

        return kinova_metrics_t


class RunType(Enum):
    REAL = 1
    SIM = 2


class JointMetricMonitor:
    """ Monitor the joint metrics to compare the range of values b/w real and sim
        Warning: Avoid except for test due to computational cost."""

    def __init__(self, device, check_point, num_robot_dofs, run_type: RunType):
        self.jp_vals = torch.empty(0, num_robot_dofs).to(device)
        self.jv_vals = torch.empty(0, num_robot_dofs).to(device)
        self.jt_vals = torch.empty(0, num_robot_dofs).to(device)
        self.run_type = run_type
        self.check_point = check_point  # this monitor can be executed only during tes
        self.num_robot_dofs = num_robot_dofs

    def append(self, jp, jv, jt):
        assert jp.shape == torch.Size([1, self.num_robot_dofs])
        assert jv.shape == torch.Size([1, self.num_robot_dofs])
        assert jt.shape == torch.Size([1, self.num_robot_dofs])

        jp = jp.detach().clone()
        jv = jv.detach().clone()
        jt = jt.detach().clone()

        self.jp_vals = torch.cat((self.jp_vals, jp), dim=0)
        self.jv_vals = torch.cat((self.jv_vals, jv), dim=0)
        self.jt_vals = torch.cat((self.jt_vals, jt), dim=0)

    @staticmethod
    def serialize(filename, _joint_monitor):
        assert _joint_monitor.run_type.name.lower() in filename, "Use the SIM/REAL runtype in the file name"
        with open(filename, 'wb') as f:
            pickle.dump(_joint_monitor, f)

    @staticmethod
    def deserialize(filename):
        with open(filename, 'rb') as f:
            _joint_monitor = pickle.load(f)
            assert _joint_monitor.run_type.name.lower() in filename, "Use the SIM/REAL runtype in the file name"
        return _joint_monitor

    def quartiles(self):
        quartile_points = torch.tensor([0., 25., 50., 75., 100.], dtype=torch.float32).to(self.jp_vals.device)
        return self._quantiles(_quantile_points=quartile_points)

    def deciles(self):
        decile_points = torch.tensor([0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.],
                                     dtype=torch.float32).to(self.jp_vals.device)
        return self._quantiles(_quantile_points=decile_points)

    def _quantiles(self, _quantile_points):
        torch.set_printoptions(precision=4, sci_mode=False)
        assert self.jp_vals.shape == self.jv_vals.shape
        assert self.jp_vals.shape == self.jt_vals.shape

        # Calculate quartiles
        jp_quartiles = torch.quantile(self.jp_vals, _quantile_points / 100, dim=0)
        jv_quartiles = torch.quantile(self.jv_vals, _quantile_points / 100, dim=0)
        jt_quartiles = torch.quantile(self.jt_vals, _quantile_points / 100, dim=0)

        # quartiles now has shape (10, 6) where each row represents a percentile
        # and each column represents a feature

        result = {
            'jp_quartiles': jp_quartiles,
            'jv_quartiles': jv_quartiles,
            'jt_quartiles': jt_quartiles
        }

        def _print_aligned_tensor(_tensor):
            for row in _tensor:
                print(" ".join(f"{value:8.2f}" for value in row))

        # Printing each tensor
        for key, tensor in result.items():
            print(f"{key}:")
            _print_aligned_tensor(tensor)

        return result

    @staticmethod
    def deviations(_joint_monitor1, _joint_monitor2):
        assert _joint_monitor1.run_type != _joint_monitor2.run_type, "comparison only between real and sim"
        sim_monitor = _joint_monitor1 if _joint_monitor1.run_type == RunType.SIM else _joint_monitor2
        real_monitor = _joint_monitor1 if _joint_monitor1.run_type == RunType.REAL else _joint_monitor2

        assert sim_monitor is not None and real_monitor is not None
        assert sim_monitor.run_type != real_monitor.run_type, "comparison only between real and sim"

        assert sim_monitor.check_point == real_monitor.check_point, "Policy should be same in real and sim."

        sim_quantiles = sim_monitor.deciles(sim_monitor)
        real_quantiles = real_monitor.deciles(real_monitor)

        result = {}

        for key in sim_quantiles.keys():
            sim_metric = torch.tensor(sim_quantiles[key])
            real_metric = torch.tensor(real_quantiles[key])
            assert sim_metric.shape == real_metric.shape

            euclidean_distance = torch.norm(sim_metric - real_metric)
            result_key = f"{key.split('_')[0]}_dist"
            result[result_key] = euclidean_distance.item()

        return result


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

    # Pos in simulation: [0.7857113649845123, 0.5376197695732117, 0.5049141645431519]
    # Rot in simulation: [0.213, -0.687, -0.61, 0.333]
    # Pos in simulation: [0.4851, -0.0555, 0.8501]
