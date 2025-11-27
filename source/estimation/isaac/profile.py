import numpy as np
from typing import List
import random
import itertools
import torch


class ForceProfile:
    """ A sequence pair of time steps and forces.
    Note: Here dt represents time steps rather than the clock time."""

    def __init__(self, ft_knots, dt_knots=[40, 40, 40]):
        self.ft_knots = ft_knots
        self.dt_knots = dt_knots

        assert len(self.dt_knots) == len(self.ft_knots), "The force knots and the dt knots should match"
        self.num_knots = len(self.dt_knots)
        self.num_steps = sum(self.dt_knots)

    @classmethod
    def from_rand_like(cls, base_fp):
        """ Create a random profile like the base force profile"""
        _rand = np.random.uniform(low=0.5, high=1.7, size=len(base_fp.dt_knots))
        rand_like = base_fp.ft_knots * _rand
        ft_knots = rand_like.round(2).tolist()
        random.shuffle(ft_knots)
        return cls(ft_knots=ft_knots)

    @classmethod
    def from_ideal_force(cls, ideal_force, num_knots=3):
        ft_knots = np.linspace(start=ideal_force * 0.8,
                               stop=ideal_force * 1.7,
                               num=num_knots).round(2).tolist()

        random.shuffle(ft_knots)
        return cls(ft_knots=ft_knots)

    @classmethod
    def from_fixed_force(cls, fixed_force, num_knots=3):
        ft_knots = np.full(shape=num_knots, fill_value=fixed_force).tolist()
        return cls(ft_knots=ft_knots)

    @staticmethod
    def gen_fixed_force_profiles(fixed_forces: List):
        return [ForceProfile.from_fixed_force(f) for f in fixed_forces]

    @staticmethod
    def profile_props(force_profiles):
        sample_fp = force_profiles[0]
        assert all(fp.num_knots == sample_fp.num_knots for fp in force_profiles) is True
        assert all(fp.dt_knots == sample_fp.dt_knots for fp in force_profiles) is True
        assert all(fp.num_steps == sample_fp.num_steps for fp in force_profiles) is True
        return sample_fp.num_steps, sample_fp.num_knots, sample_fp.dt_knots

    def __repr__(self):
        return f'ForceProfile(dt_knots={self.dt_knots}, ft_knots={self.ft_knots})'


class Experiment:
    """ Represents a true physical experiment that defines the force profile-trajectory relationships.
        Can be measured from Physical world or from simulation itself.
        Noise can be added in for robustness case of simulated exp. """

    def __init__(self, link_name, force_profile: ForceProfile, true_pos_trajectory, true_vel_trajectory,
                 true_force_trajectory=None, noise=None, time_trajectory=None, mc=None, grasp_loc=0.5):
        self.link_name = link_name
        self.force_profile = force_profile  # The applied force

        # Measured forces: in real world experiments, the applied force may differ from end-effector force
        self.true_force_trajectory = true_force_trajectory

        self.true_pos_trajectory = true_pos_trajectory
        self.true_vel_trajectory = true_vel_trajectory

        # measurement constants, instance of Measurement Constants class dict
        self.mc = mc

        if noise:  # Noise should not be added for real world experiments, due to inherent noise from nature
            self.inject_noise(noise)

        self.estimated_pos_trajectories = None
        self.estimated_vel_trajectories = None

        # deviation b/w true and estimated trajectories
        self.avg_pos_deviation = None

        # in real world experiments, the time instances may not be evenly spaced
        self.time_trajectory = time_trajectory

        self.grasp_loc = grasp_loc  # location to grasp the branch and apply force
        self.validate_traj()

    def inject_noise(self, noise):
        self.noise = noise

        # a single noisy pos/velocity trajectory
        self.noisy_pos_trajectory = self.noise(self.true_pos_trajectory)
        self.noisy_vel_trajectory = self.noise(self.true_vel_trajectory)

        # a single noisy force profile trajectory
        self.noisy_force_profile = self.noise_fp(self.force_profile, self.noise, self.true_pos_trajectory.device)

    @staticmethod
    def noise_fp(force_profile, noise, device):

        num_steps, dt_knots = force_profile.num_steps, force_profile.dt_knots
        dilated_dt_knots = [1 for _ in range(num_steps)]

        dilated_ft_knots_sep = [[force_profile.ft_knots[idx]] * dt for idx, dt in enumerate(dt_knots)]
        dilated_ft_knots = list(itertools.chain(*dilated_ft_knots_sep))

        assert len(dilated_ft_knots) == num_steps

        dilated_ft_knots_t = torch.FloatTensor(dilated_ft_knots).to(device)
        return ForceProfile(ft_knots=noise(dilated_ft_knots_t).tolist(), dt_knots=dilated_dt_knots)

    def set_estimated(self, estimated_pos_trajectories, estimated_vel_trajectories, avg_pos_deviation):
        self.estimated_pos_trajectories = estimated_pos_trajectories
        self.estimated_vel_trajectories = estimated_vel_trajectories
        self.avg_pos_deviation = avg_pos_deviation

    def validate_traj(self):
        zero_state_offset = 0 if self.mc is None else self.mc['zero_state_offset']
        traj_length = self.true_pos_trajectory.shape[0] if self.force_profile is None \
            else self.force_profile.num_steps + zero_state_offset

        assert traj_length == self.true_pos_trajectory.shape[0], "Bad Profile"
        assert self.true_vel_trajectory is None or traj_length == self.true_vel_trajectory.shape[0], "Bad Profile"
        assert self.true_force_trajectory is None or traj_length == self.true_force_trajectory.shape[0], "Bad Profile"
        assert self.time_trajectory is None or traj_length == self.time_trajectory.shape[0], "Bad Profile"

        if self.grasp_loc is not None:
            assert 0. < self.grasp_loc <= 1.0, "Invalid grasp. Should be > 0.& <= 1.0, : (dist frm origin)/(branch len)"
