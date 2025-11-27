import sys
from typing import List
from itertools import chain
import numpy as np
from isaacgym import gymapi
from isaacgym import gymtorch

from estimation.isaac.param import Param, JointParams, StiffnessType, DampingType
from estimation.math import common
from estimation.isaac.profile import ForceProfile, Experiment
import torch


class Mechanics:
    def __init__(self, gym, sim, dt, sub_steps, device, stepper, gym_state, artifact_state, logging):
        self.gym = gym
        self.sim = sim
        self.dt = dt
        self.sub_steps = sub_steps

        self.device = device

        self.stepper = stepper
        self.gym_state = gym_state
        self.artifact_state = artifact_state

        self.num_envs = self.gym_state.num_envs
        self.num_bodies = self.gym_state.num_bodies
        self.dofs_per_tree_per_env = self.gym_state.dofs_per_tree_per_env

        self.logging = logging
        self.force_axis = 2  # only z for now

        # Note: isaacgym/docs/programming/tuning.html?highlight=performance#common-parameters
        # assert round(self.dt / self.sub_steps, 2) <= 1 / 50., "Stable PhysX sim requires ts ,under 1/50th of 1 sec"

    def wait_for_branch_settling(self, branch_settlement_tolerance):
        """
        To ensure branches (across environments) have settled down from the previous shake.
        Raise alarm if not settled after max steps. Velocity should be close but won't be exact 0, due to gravity e.t.c
        """
        max_wait_steps = 1500
        sim_dt_steps_per_wait = 3
        # For e.g. 1.01 means, if the change in velocity/pos from last sim step is under 1%, the branches are ~ at rest.
        self.logging.warn(f"Warning : Current branch settling tolerance is {branch_settlement_tolerance * 100}%")

        for wait_steps in range(sys.maxsize):
            self.stepper.step_through(steps=sim_dt_steps_per_wait)
            self.gym.refresh_dof_state_tensor(self.sim)
            worst_branch_vel = torch.max(torch.abs(self.artifact_state.dof_states[:, 1]))
            worst_branch_pos = torch.max(torch.abs(self.artifact_state.dof_states[:, 0]))

            if wait_steps == 0:
                prev_worst_branch_vel = worst_branch_vel
                prev_worst_branch_pos = worst_branch_pos
                self.logging.debug(f"Skipping iteration:  Branch Vel : {worst_branch_vel}, Pos : {worst_branch_pos}")
                continue

            #  done if change in velocity/pos is < 0.001 * prev value
            change_in_pos = worst_branch_pos - prev_worst_branch_pos
            change_in_velocity = worst_branch_vel - prev_worst_branch_vel

            total_steps = wait_steps * sim_dt_steps_per_wait
            time_taken = wait_steps * sim_dt_steps_per_wait * self.dt

            if torch.lt(torch.abs(change_in_pos), prev_worst_branch_pos * branch_settlement_tolerance):
                if torch.lt(torch.abs(change_in_velocity), prev_worst_branch_vel * branch_settlement_tolerance):
                    self.logging.info(f"""Finished Wait to settle: Vel : {worst_branch_vel},\
                     Pos : {worst_branch_pos}, Steps: {total_steps}, Time:{time_taken} sec""".replace("  ", ""))
                    break

            self.logging.info(
                f"""Wait to settle : Vel: {worst_branch_vel} Pos:{worst_branch_pos},
                 Steps: {total_steps}, Time:{time_taken} sec""")

            prev_worst_branch_vel = worst_branch_vel
            prev_worst_branch_pos = worst_branch_pos

            if wait_steps >= max_wait_steps:
                msg = f"Branches haven't settled : Worst Branch Vel : {worst_branch_vel}, Waited for ({time_taken} sec"
                self.logging.error(msg)
                raise ValueError(msg)

        if time_taken > 50.:
            print(f"Long wait : Vel: {worst_branch_vel} Pos:{worst_branch_pos}, Steps: {total_steps}, T:{time_taken} ")

    def generate_stable_states(self, tree_handle, dof_id, p_damping_val, param_base_len=300, lower=1., upper=300.):
        """  Generate a dictionary of stable states for stiffness parameters for future resets.
        Dict contains the settled state of the specific dof for each stiffness value.
        It is significantly faster to reset to this stable state rather than to zero,
        """

        sim_p_stiffness = Param.from_range(count=param_base_len, lower=lower, upper=upper, param_type=StiffnessType())
        sim_p_damping = Param.from_fixed_val(count=param_base_len, value=p_damping_val, param_type=DampingType())

        # here combination of stiffness/damping is not required
        joint_params = [JointParams(dof_id=dof_id, jp_stiffness=sim_p_stiffness, jp_damping=sim_p_damping)]
        target_dofs = [jp.dof_id for jp in joint_params]

        root_stiffness_vals, root_damping_vals = [], []

        stable_states = []
        while True:

            distr_p_stiffness = {
                jp.dof_id: jp.jp_stiffness.next_slice(curr_idx=len(root_stiffness_vals), slice_length=self.num_envs)
                for jp in joint_params}
            root_stiffness_vals.extend(distr_p_stiffness.get(target_dofs[0]))

            distr_p_damping = {
                jp.dof_id: jp.jp_damping.next_slice(curr_idx=len(root_damping_vals), slice_length=self.num_envs)
                for jp in joint_params}
            root_damping_vals.extend(distr_p_damping.get(target_dofs[0]))

            self.gym_state.set_distr_tree_params(tree_handle=tree_handle,
                                                 target_dofs=target_dofs,
                                                 distr_p_stiffness=distr_p_stiffness,
                                                 distr_p_damping=distr_p_damping,
                                                 default_p_stiffness_val=StiffnessType().default_val,
                                                 default_p_damping_val=DampingType().default_val)
            self.reset_to_zero()
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            stable_dof_whole = self.artifact_state.dof_states.detach().clone()

            stable_dof_states_by_env = torch.split(stable_dof_whole,
                                                   split_size_or_sections=self.gym_state.dofs_per_tree_per_env)

            stable_states.extend([ss[dof_id] for ss in stable_dof_states_by_env])

            if len(root_stiffness_vals) >= joint_params[0].jp_stiffness.param_len() or root_stiffness_vals[-1] == 0.:
                break

        # remove all invalid parameters at the end
        while root_stiffness_vals[-1] == 0. or root_stiffness_vals[-1] == Param.some_max_param_value:
            root_stiffness_vals = root_stiffness_vals[:-1]
            root_damping_vals = root_damping_vals[:-1]

        assert len(root_stiffness_vals) == param_base_len, "Error in stable state dict generation"

        return {st: stable_state for st, stable_state in zip(root_stiffness_vals, stable_states)}

    def reset_to_zero(self):
        """ Reset the d.o.fs of all environments to zero. All actors d.o.fs will be reset """
        self.logging.info("Reset to Zero State ....")
        zero_pos_dof_states = torch.zeros_like(self.artifact_state.dof_states)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(zero_pos_dof_states))
        self.wait_for_branch_settling(branch_settlement_tolerance=0.01)

    def hard_reset(self, tree_handle):
        """ Set all parameters to defaults values and then reset the d.o.fs of all environments to zero. """
        for e in range(self.num_envs):
            p_stiffness_vals = np.full(shape=self.gym_state.dofs_per_tree_per_env,
                                       fill_value=StiffnessType().default_val,
                                       dtype=np.float32)
            p_damping_vals = np.full(shape=self.gym_state.dofs_per_tree_per_env,
                                     fill_value=DampingType().default_val,
                                     dtype=np.float32)
            self.gym_state.set_params_by_env(e, tree_handle, p_stiffness_vals, p_damping_vals, p_friction=0.0)

        self.reset_to_zero()
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def reset_to_specific_state(self, specific_dof_states):
        """ Reset the d.o.fs of sll environments to a specific state. All actors d.o.fs will be reset """
        self.logging.info("Reset to a Specific State ....")
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(specific_dof_states))
        self.wait_for_branch_settling(branch_settlement_tolerance=0.01)

    def reset_to_stable_state(self, distr_p_stiffness, all_stable_states):
        """ Reset all d.o.fs of all environments to corresponding stable states.
         All non-measured branches/dofs will have zero state
         All measured branches/dofs will be set to a stable state depending on the stiffness parameter value. """

        def _nearest_stable_dof_state(p_val, stable_state_dict):
            """currently stable_state dict does not contain decimals, hence round to reach the nearest parameter """
            nearest_p_val = 1 if p_val <= 1. else float(round(p_val))
            if nearest_p_val not in stable_state_dict.keys():
                if p_val == Param.some_max_param_value or p_val == StiffnessType().default_val:
                    return torch.zeros_like([*stable_state_dict.values()][0])
                raise ValueError(f"Nearest Value not available in stable dictionary: {p_val}")
            return stable_state_dict[nearest_p_val]

        self.logging.info("Reset to Stable State ....")
        zero_stable_dof_states = torch.zeros_like(self.artifact_state.dof_states)

        assert len(all_stable_states) == len(
            distr_p_stiffness), "Count of stable state dictionaries should match the stiffness parameters."

        # TODO: This fn may be improved by replacing the dict with a tensor that removes the for loop
        # TODO: The below function will work only in cases where there is a single body in the environment
        for dof_id, p_stiffness in distr_p_stiffness.items():
            br_stable_state = all_stable_states[dof_id]
            for e in range(self.num_envs):
                zero_stable_dof_states[
                    e * self.dofs_per_tree_per_env + dof_id] = _nearest_stable_dof_state(p_stiffness[e],
                                                                                         br_stable_state)

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(zero_stable_dof_states))
        self.wait_for_branch_settling(branch_settlement_tolerance=0.2)

    def generate_distr_ft_tensor(self, external_force_vals_t, link_id):
        """
        Given a tensor of forces, generate a FT tensor for each combo
        Each env in the tensor will have a separate force attached ti it.
        Input: {external_force_vals_t} tensor:  shape  (num_knots x num_envs)
        Output: force tensor: shape = [num_knots x num_envs x num_bodies x 3]
        Output: torque tensor: shape = [num_knots x num_envs x num_bodies x 3], with zeros
        """

        num_knots = external_force_vals_t.shape[0]

        assert external_force_vals_t.shape[1] == self.num_envs, "Invalid force tensor shape"

        forces = torch.zeros((num_knots, self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # torque is zero for this project
        torques = torch.zeros((num_knots, self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        # The -1 is `for downward in the world frame
        forces[:, :, link_id, self.force_axis] = external_force_vals_t * -1

        return forces, torques

    @staticmethod
    def mean_deformation_force(ft, deformation):
        """ Given a force and corresponding deformation array,
         finds the force that generates the mean deformation"""

        if len(ft) != len(deformation):
            raise ValueError("Invalid Array lengths")

        mean_deformation_boundary = max(deformation) / 2.

        for _mx, _my in zip(ft, deformation):
            if _my >= mean_deformation_boundary:
                return round(_mx, 2), round(_my, 4)

    def apply_force_profile(self, tree_handle, link_offsets, link_child_offsets, force_profiles: List[ForceProfile],
                            grasp_locs, space=gymapi.ENV_SPACE):
        """
        Apply an external force in the world frame, at the c.o.m of the branch,
        currently just a pull down perpendicular to the ground .
        Return a deformation trajectory for position and velocity.

        tree_handle : actor handle e.g. 0 for ground truth, 1 for sim, e.t.c.
        Currently, Deformation is measured as
        Positional Change : Euclidean distance b/w start and end position at c.o.m
        Velocity Change: Change in Linear Velocity (not angular) of the link at c.o.m
        space = ENV_SPACE or LOCAL_SPACE, Use local space to apply forces perpendicular to the body
        """

        self.logging.debug(f"Apply Force & Generate a deformation trajectory for tree: ...{tree_handle}")

        assert link_offsets is not None and len(link_offsets) == self.num_envs, 'Link to apply force is missing'
        assert grasp_locs is not None and len(grasp_locs) == self.num_envs, 'Invalid set of grasp_locs'

        link_id = link_offsets[0]
        grasp_loc_t = self.gym_state.grasp_loc_t(grasp_locs).to(self.device)

        # it is possible that this start position varies b/w iterations
        link_world_pos_start = self.gym_state.link_world_position_at_dist(
            self.artifact_state.rb_positions, link_offsets, link_child_offsets, grasp_loc_t)

        link_world_lin_vel_start = self.gym_state.link_world_lin_vel_at_dist(
            self.artifact_state.rb_linvels, link_offsets, link_child_offsets, grasp_loc_t)

        self.logging.debug(f"link_world_pos_start: {link_world_pos_start}")
        self.logging.debug(f"link_world_lin_vel_start: {link_world_lin_vel_start}")

        #  [[f11,f21,f31...f_num_env1], [f12,f22,f32...f_num_env2], ..... num of knots]
        num_steps, num_knots, dt_knots = ForceProfile.profile_props(force_profiles)

        force_vals_x_knots = [[fp.ft_knots[idx] for fp in force_profiles]
                              for idx, _ in enumerate(dt_knots)]
        assert len(force_vals_x_knots) == num_knots
        assert len(force_vals_x_knots[0]) == self.num_envs

        force_vals_x_knots_t = torch.tensor(force_vals_x_knots, device=self.device)
        forces, torques = self.generate_distr_ft_tensor(external_force_vals_t=force_vals_x_knots_t, link_id=link_id)

        # Apply external force for a short duration & measure the shift of the c.o.m pos & vel
        pos_deform_states, vel_deform_states = torch.empty(0, device=self.device), torch.empty(0, device=self.device)

        force_not_at_com = any([_l != 0.5 for _l in grasp_locs])  # is the force to be applied at com or elsewhere
        self.logging.debug(f"Is force applied at com: {force_not_at_com}")

        for idx, dt in enumerate(dt_knots):
            for _ in range(dt):
                if force_not_at_com:
                    force_positions = torch.zeros_like(self.artifact_state.rb_positions)
                    link_force_positions = self.gym_state.link_world_position_at_dist(
                        self.artifact_state.rb_positions, link_offsets, link_child_offsets, grasp_loc_t)
                    force_positions[link_offsets, :] = link_force_positions
                    self.gym.apply_rigid_body_force_at_pos_tensors(sim=self.sim,
                                                                   forceTensor=gymtorch.unwrap_tensor(forces[idx]),
                                                                   posTensor=gymtorch.unwrap_tensor(force_positions),
                                                                   space=space)
                else:
                    # Apply forces at center of mass,
                    # same as above when grasp loc is 0.5, but this is slightly more accurate
                    self.gym.apply_rigid_body_force_tensors(sim=self.sim,
                                                            forceTensor=gymtorch.unwrap_tensor(forces[idx]),
                                                            torqueTensor=gymtorch.unwrap_tensor(torques[idx] * 0),
                                                            space=space)

                self.stepper.step_through(steps=1)

                self.gym.refresh_rigid_body_state_tensor(self.sim)
                link_world_pos_end = self.gym_state.link_world_position_at_dist(
                    self.artifact_state.rb_positions, link_offsets, link_child_offsets, grasp_loc_t)

                link_world_lin_vel_end = self.gym_state.link_world_lin_vel_at_dist(
                    self.artifact_state.rb_linvels, link_offsets, link_child_offsets, grasp_loc_t)

                curr_pos_deformations = common.l2_norm(link_world_pos_end, link_world_pos_start)
                curr_lin_vel_deformations = common.l2_norm(link_world_lin_vel_end, link_world_lin_vel_start)

                self.logging.debug(f"curr_pos_deformations: {curr_pos_deformations.shape}, {curr_pos_deformations}")

                # store the deformation states to build a trajectory later on
                pos_deform_states = torch.cat((pos_deform_states, curr_pos_deformations.unsqueeze(0)), dim=-2)

                vel_deform_states = torch.cat((vel_deform_states, curr_lin_vel_deformations.unsqueeze(0)), dim=-2)

        assert pos_deform_states.shape == torch.Size([num_steps, self.num_envs]), "Invalid Traj Shape"
        pos_trajectories = torch.transpose(pos_deform_states, 0, 1)

        assert vel_deform_states.shape == torch.Size([num_steps, self.num_envs]), "Invalid Traj Shape"
        vel_trajectories = torch.transpose(vel_deform_states, 0, 1)

        external_force_duration = round(num_steps * self.dt, 2)

        self.logging.debug(f"""Applying: ", {torch.FloatTensor(force_vals_x_knots)}, " on link:", {link_id},
         for duration(sec):", {external_force_duration}, "  Max Pos Deformation: ", 
        {self.get_deforms_max(pos_trajectories)}, "  Sum Deformation: ",{self.get_deforms_sum(pos_trajectories)}""")

        return pos_trajectories, vel_trajectories

    @staticmethod
    def get_deforms_max(trajectories):
        out, _ = torch.max(trajectories, dim=1)
        return out

    @staticmethod
    def get_deforms_mean(trajectories):
        return torch.mean(trajectories, dim=1)

    @staticmethod
    def get_deforms_sum(trajectories):
        return torch.sum(trajectories, dim=1)

    @staticmethod
    def get_deforms_steady(trajectories):
        return trajectories[:, - 1]

    def cost_pos(self, sim_pos_deforms, gt_ideal_pos_change):
        """ Given deformation tensors & ideal deformation calculate the cost;  i.e.,
        the deviation of change(either pos change(deformation) or change in velocity or their sum) from the ground truth

        E.g. cost(st) = L2-norm { gt deformation (fi), sim deformation (fi) }
        st=stiffness, fi is the ideal force that gives the mean deformation in gt.
        """

        self.logging.debug(f"Computing cost profile against ideal pos change: {gt_ideal_pos_change}")
        return common.l2_norm(
            sim_pos_deforms.unsqueeze(1),
            torch.zeros_like(sim_pos_deforms, device=self.device).fill_(gt_ideal_pos_change).unsqueeze(1))

    def cost_sum_profile(self, sim_pos_deforms_sum, **def_kwargs):
        """  Uses the sum profile to measure the deformations  """
        gt_ideal_sum_pos_change = def_kwargs['gt_ideal_sum_pos_change']
        self.logging.debug(f"Taking sum cost profile against ideal pos change: {gt_ideal_sum_pos_change}")
        return self.cost_pos(sim_pos_deforms_sum, gt_ideal_sum_pos_change)

    def cost_max_profile(self, sim_pos_deforms_max, **def_kwargs):
        """ Uses the max profile to measure the deformations """
        gt_ideal_max_pos_change = def_kwargs['gt_ideal_max_pos_change']
        self.logging.debug(f"Taking max cost profile against ideal pos change: {gt_ideal_max_pos_change}")
        return self.cost_pos(sim_pos_deforms_max, gt_ideal_max_pos_change)

    def cost_steady_profile(self, sim_pos_deforms_steady, **def_kwargs):
        """ Uses the max profile to measure the deformations """
        gt_ideal_steady_pos_change = def_kwargs['gt_ideal_steady_pos_change']
        self.logging.debug(f"Taking steady cost profile against ideal pos change: {gt_ideal_steady_pos_change}")
        return self.cost_pos(sim_pos_deforms_steady, gt_ideal_steady_pos_change)

    def gt_deformations(self, gt_tree_handle, min_force_delta, max_force, link_offsets, link_child_offsets,
                        initial_dof_states, grasp_loc):
        """ Create the deformation matrix for the ground truth
        Start from {min_force_delta} to {max_force}, increment by {min_force_delta}
        link_offsets & link_child_offsets is to find the pos of c.o.m the link after deformation
        link_com_world_pos_init: refers to the c.o.m pos of the link before any force is applied
        initial_dof_states: dof positions before any force is applied, after true params are applied

        This method is only used in simulations to create a ground truth to capture the ideal force
        """

        def _next_slice(curr, _num_envs):
            _slice = [curr + (i + 1) * min_force_delta for i in range(_num_envs)]
            return [_s if _s <= max_force else 0. for _s in _slice]

        ft = []
        gt_pos_trajectories = torch.empty(0, device=self.device)
        gt_vel_trajectories = torch.empty(0, device=self.device)

        while True:
            if len(ft) == 0:
                nxt_ft_slice = _next_slice(0., self.num_envs)
                ft.extend(nxt_ft_slice)
            else:
                nxt_ft_slice = _next_slice(ft[-1], self.num_envs)
                ft.extend(nxt_ft_slice)

            self.reset_to_specific_state(initial_dof_states)
            nxt_ft_slice_profiles = ForceProfile.gen_fixed_force_profiles(nxt_ft_slice)
            pos_trajectories, vel_trajectories = self \
                .apply_force_profile(tree_handle=gt_tree_handle,
                                     link_offsets=link_offsets,
                                     link_child_offsets=link_child_offsets,
                                     force_profiles=nxt_ft_slice_profiles,
                                     grasp_locs=[grasp_loc for _ in range(self.num_envs)])

            gt_pos_trajectories = torch.cat((gt_pos_trajectories, pos_trajectories), dim=0)
            gt_vel_trajectories = torch.cat((gt_vel_trajectories, vel_trajectories), dim=0)

            self.reset_to_specific_state(initial_dof_states)

            if ft[-1] >= max_force or ft[-1] == 0.:
                break

        assert len(ft) == gt_pos_trajectories.shape[0]
        assert len(ft) == gt_vel_trajectories.shape[0]

        # remove all invalid ft at the end
        while ft[-1] == 0.:
            ft = ft[:-1]
            gt_pos_trajectories = gt_pos_trajectories[:-1]
            gt_vel_trajectories = gt_vel_trajectories[:-1]

        return ft, gt_pos_trajectories, gt_vel_trajectories

    def validate_joint_params(self, joint_params: List[JointParams]):
        target_dofs = [jp.dof_id for jp in joint_params]

        self.logging.debug(f"Sim Deformations joint_params: {joint_params}")
        assert len(joint_params) <= 3, "Currently joint param inference implemented only for " \
                                       "exactly 1,2 or 3 sequential branches."

        # check all parameter lengths across stiffness/damping/different branches are exactly equal
        assert len(set([jp.param_len() for jp in joint_params])) == 1

        # target dof should always be upward from trunk
        if len(joint_params) >= 2:
            assert all(target_dofs[i] <= target_dofs[i + 1] for i in range(len(target_dofs) - 1)), "Wrong branch order"

    def validate_def_kwargs(self, kwargs):
        self.logging.debug(f"Sim Deformations Kwargs: {kwargs}")

        for kw in kwargs:
            self.gym_state.validate_parent_child(kw.link_offsets, kw.link_child_offsets)

        # nothing to validate
        if len(kwargs) == 1:
            return

        branch_names = [kw.link_name for kw in kwargs]
        # if all deformation kwargs are for the same branch, it is OK
        if all(x == branch_names[0] for x in branch_names) is True:
            return

        self.logging.warn(f"Poor performance can be caused by using trajectories from multiple branches: {len(kwargs)}")

        # To do this across multiple trajectories link_offsets and  link_child_offsets across kwargs must be combined.
        assert all([kw.link_offsets == kwargs[0].link_offsets for kw in
                    kwargs]) is True, " Currently multiple trajectories are only implemented for the same branch shake."

        assert all([kw.link_child_offsets == kwargs[0].link_child_offsets for kw in
                    kwargs]) is True, " Currently multiple trajectories are only implemented for the same branch shake"

        # If branches are not same, then they should be connected
        if len(kwargs) == 2:
            outer_branch_name = branch_names[-1]
            inner_branch_id = outer_branch_name[outer_branch_name.find('P'):].replace('P', 'B')
            assert inner_branch_id in branch_names[-2], "Only Connected branches (need not be parent/child) can be used"
        else:
            raise ValueError("The Validation for more than 2 branches is not implemented")

    def sim_multi_deformations(self, sim_tree_handle, joint_params: List[JointParams], all_stable_states, kwargs):
        """ Given parameter objects & ideal ground truth forces for coupled branches calculate the deformation cost;
          i.e., cost is calculated w.r.t to the true position and velocity trajectories from the ground truth.

        kw: deformation kwargs that contain link_offsets, link_child_offsets, link_com_world_pos_init etc.
        E.g. Start from {min_stiffness} to {max_stiffness}, increment by stiffness_delta
        link_offsets & link_child_offsets is to find the pos of c.o.m the link after deformation
        link_com_world_pos_init: refers to the c.o.m pos of the link before any force is applied """

        # target_dofs: dofs where the parameters are applied (e.g [sim_parent1_dof_id, sim_child_dof_id]
        target_dofs = [jp.dof_id for jp in joint_params]

        self.validate_joint_params(joint_params)
        self.validate_def_kwargs(kwargs)

        num_train_trajectories = len(kwargs)

        total_deform_cost = torch.empty(0, device=self.device)
        slice_deform_cost = None

        root_stiffness_vals, root_damping_vals = [], []

        while True:
            # parameters are repeated for each trajectory in training
            param_repeat_len = int(self.num_envs / num_train_trajectories)
            index_length = int(len(root_stiffness_vals) / num_train_trajectories)

            distr_p_stiffness = {
                jp.dof_id: jp.jp_stiffness.next_slice(curr_idx=index_length,
                                                      slice_length=param_repeat_len) * num_train_trajectories
                for jp in joint_params}
            root_stiffness_vals.extend(distr_p_stiffness.get(target_dofs[0]))

            distr_p_damping = {
                jp.dof_id: jp.jp_damping.next_slice(curr_idx=index_length,
                                                    slice_length=param_repeat_len) * num_train_trajectories
                for jp in joint_params}
            root_damping_vals.extend(distr_p_damping.get(target_dofs[0]))

            if len(joint_params) >= 2:
                assert all(len(distr_p_stiffness[target_dofs[i]]) == len(distr_p_stiffness[target_dofs[i + 1]])
                           for i in range(len(target_dofs) - 1)), "Invalid Kp length"

                assert all(len(distr_p_damping[target_dofs[i]]) == len(distr_p_damping[target_dofs[i + 1]])
                           for i in range(len(target_dofs) - 1)), "Invalid Kd length"

            self.gym_state \
                .set_distr_tree_params(tree_handle=sim_tree_handle,
                                       target_dofs=target_dofs,
                                       distr_p_stiffness=distr_p_stiffness,
                                       distr_p_damping=distr_p_damping,
                                       default_p_stiffness_val=StiffnessType().default_val,
                                       default_p_damping_val=DampingType().default_val)

            # shake same branch with different force/trajectories according to the kwargs;
            # but for the same parameter set.
            self.reset_to_zero() if all_stable_states is None else \
                self.reset_to_stable_state(distr_p_stiffness, all_stable_states)

            # at the moment, only same branch can be used for different trajectories, so anyone can be picked
            comb_link_offsets = kwargs[0].link_offsets
            comb_link_child_offsets = kwargs[0].link_child_offsets

            # combine force profiles, offsets if multiple trajectories are used for training
            comb_force_profiles = list(chain(*[kw.force_profiles for kw in kwargs]))
            comb_grasp_locs = list(chain(*[kw.grasp_locs for kw in kwargs]))
            ignore_velocity_cost = any([kw.ignore_velocity_cost for kw in kwargs])

            pos_trajectories, vel_trajectories = self \
                .apply_force_profile(tree_handle=sim_tree_handle,
                                     link_offsets=comb_link_offsets,
                                     link_child_offsets=comb_link_child_offsets,
                                     force_profiles=comb_force_profiles,
                                     grasp_locs=comb_grasp_locs)

            _comb_gt_ideal_pos_trajectory = torch.cat(tuple([kw.true_pos_trajectory.unsqueeze(0) \
                                                            .repeat_interleave(param_repeat_len, dim=0)
                                                             for kw in kwargs]), dim=0)

            _comb_gt_ideal_vel_trajectory = torch.cat(tuple([kw.true_vel_trajectory.unsqueeze(0) \
                                                            .repeat_interleave(param_repeat_len, dim=0)
                                                             for kw in kwargs]), dim=0)

            # cost is computed as sum of pos trajectory deviation & vel trajectory deviation
            if ignore_velocity_cost is True:
                temp_deform_cost = common.l2_norm(pos_trajectories, _comb_gt_ideal_pos_trajectory)
            else:

                temp_deform_cost = (common.l2_norm(pos_trajectories, _comb_gt_ideal_pos_trajectory)
                                    + common.l2_norm(vel_trajectories, _comb_gt_ideal_vel_trajectory))

            slice_deform_cost = temp_deform_cost if slice_deform_cost is None \
                else slice_deform_cost + temp_deform_cost

            total_deform_cost = torch.cat((total_deform_cost, slice_deform_cost), dim=0)

            if len(root_stiffness_vals) >= self.num_envs or root_stiffness_vals[-1] == 0.:
                break

        # remove all invalid parameters at the end
        while root_stiffness_vals[-1] == 0. or root_stiffness_vals[-1] == Param.some_max_param_value:
            root_stiffness_vals = root_stiffness_vals[:-1]
            root_damping_vals = root_damping_vals[:-1]
            total_deform_cost = total_deform_cost[:-1]

        assert root_stiffness_vals == joint_params[0].jp_stiffness.param_vals.tolist() * num_train_trajectories
        assert root_damping_vals == joint_params[0].jp_damping.param_vals.tolist() * num_train_trajectories

        # deformation cost across all trajectories
        total_deform_cost = total_deform_cost.unsqueeze(1)

        # split by trajectories
        deform_cost_per_trajectory = torch.split(total_deform_cost, split_size_or_sections=param_repeat_len)

        # sum deformation cost of each training trajectory
        return torch.stack(deform_cost_per_trajectory, dim=0).sum(dim=0)

    def evaluate_params(self, sim_tree_handle, joint_params: List[JointParams], all_stable_states, kwargs):
        """ Given parameter objects, force profiles, e.t.c.,  compare the estimated trajectories to the truth"""

        # target_dofs: dofs where the parameters are applied (e.g [sim_parent1_dof_id, sim_child_dof_id]
        target_dofs = [jp.dof_id for jp in joint_params]
        self.validate_joint_params(joint_params)
        self.validate_def_kwargs(kwargs)

        assert len(kwargs) == 1, "For Evaluation, only one branch trajectory can be used at a time"
        kw = kwargs[0]

        total_avg_pos_deviation = torch.empty(0, device=self.device)
        root_stiffness_vals, root_damping_vals = [], []

        while True:

            distr_p_stiffness = {
                jp.dof_id: jp.jp_stiffness.next_slice(curr_idx=len(root_stiffness_vals), slice_length=self.num_envs)
                for jp in joint_params}
            root_stiffness_vals.extend(distr_p_stiffness.get(target_dofs[0]))

            distr_p_damping = {
                jp.dof_id: jp.jp_damping.next_slice(curr_idx=len(root_damping_vals), slice_length=self.num_envs)
                for jp in joint_params}
            root_damping_vals.extend(distr_p_damping.get(target_dofs[0]))

            if len(joint_params) >= 2:
                assert all(len(distr_p_stiffness[target_dofs[i]]) == len(distr_p_stiffness[target_dofs[i + 1]])
                           for i in range(len(target_dofs) - 1)), "Invalid Kp length"

                assert all(len(distr_p_damping[target_dofs[i]]) == len(distr_p_damping[target_dofs[i + 1]])
                           for i in range(len(target_dofs) - 1)), "Invalid Kd length"

            self.gym_state \
                .set_distr_tree_params(tree_handle=sim_tree_handle,
                                       target_dofs=target_dofs,
                                       distr_p_stiffness=distr_p_stiffness,
                                       distr_p_damping=distr_p_damping,
                                       default_p_stiffness_val=StiffnessType().default_val,
                                       default_p_damping_val=DampingType().default_val)

            self.reset_to_zero() if all_stable_states is None else \
                self.reset_to_stable_state(distr_p_stiffness, all_stable_states)

            pos_trajectories, vel_trajectories = self \
                .apply_force_profile(tree_handle=sim_tree_handle,
                                     link_offsets=kw.link_offsets,
                                     link_child_offsets=kw.link_child_offsets,
                                     force_profiles=kw.force_profiles,
                                     grasp_locs=kw.grasp_locs)

            _true_pos_trajectory_ex = kw.true_pos_trajectory.unsqueeze(0) \
                .repeat_interleave(pos_trajectories.shape[0], dim=0)

            # the average linear distance from true trajectory to the estimated trajectory
            slice_avg_pos_deviation = torch.mean(
                torch.abs(pos_trajectories - _true_pos_trajectory_ex), dim=1)
            total_avg_pos_deviation = torch.cat((total_avg_pos_deviation, slice_avg_pos_deviation), dim=0)

            if len(root_stiffness_vals) >= joint_params[0].jp_stiffness.param_len() or root_stiffness_vals[-1] == 0.:
                break

        # remove all invalid parameters at the end
        while root_stiffness_vals[-1] == 0. or root_stiffness_vals[-1] == Param.some_max_param_value:
            root_stiffness_vals = root_stiffness_vals[:-1]
            root_damping_vals = root_damping_vals[:-1]
            total_avg_pos_deviation = total_avg_pos_deviation[:-1]
            pos_trajectories = pos_trajectories[:-1, :]
            vel_trajectories = vel_trajectories[:-1, :]

        assert np.array_equal(root_stiffness_vals, joint_params[0].jp_stiffness.param_vals)
        assert np.array_equal(root_damping_vals, joint_params[0].jp_damping.param_vals)

        return total_avg_pos_deviation, pos_trajectories, vel_trajectories

    def gen_gt_experiments(self, gt_tree_handle, num_experiments, ideal_force, kw) -> List[Experiment]:
        """ Using the ideal force, first generate a set of force profiles & then corresponding pos/vel trajectories """
        experiments = []

        noise = common.GaussianNoise(device=self.device) if kw.use_noisy_profile else None

        all_grasp_locs = (np.linspace(start=0.2, stop=0.8, num=7).round(2)
                          if kw.change_grasp_locs else np.repeat(0.5, 7).round(2))

        for _ in range(num_experiments):
            grasp_loc_i = np.random.choice(all_grasp_locs, 1).item()  # choose one location to grasp for the experiment
            # force at com is the ideal force, increase on moving to parent fork & decrease on moving to child fork
            grasp_force_factor = (1 / grasp_loc_i) * 0.5
            force_profile_base = ForceProfile.from_ideal_force(ideal_force=grasp_force_factor * ideal_force)
            self.logging.debug(f"force_profile_base: {force_profile_base}")

            force_profile_i = ForceProfile.from_rand_like(base_fp=force_profile_base)
            self.logging.debug(f"Using force Profile to create an experiment: {force_profile_i}")
            self.reset_to_specific_state(kw.initial_dof_states)

            profile_pos_trajectories, profile_vel_trajectories = self \
                .apply_force_profile(tree_handle=gt_tree_handle,
                                     link_offsets=kw.link_offsets,
                                     link_child_offsets=kw.link_child_offsets,
                                     force_profiles=[force_profile_i for _ in range(self.num_envs)],
                                     grasp_locs=[grasp_loc_i for _ in range(self.num_envs)])

            # we don't need other trajectories
            profile_pos_traj, profile_vel_traj = profile_pos_trajectories[0], profile_vel_trajectories[0]

            exp = Experiment(link_name=kw.link_name,
                             force_profile=force_profile_i,
                             true_pos_trajectory=profile_pos_traj,
                             true_vel_trajectory=profile_vel_traj,
                             noise=noise,
                             grasp_loc=grasp_loc_i)
            experiments.append(exp)

        return experiments


class DeformationKwargs:
    """ Arguments required to run simulated trees and estimate the cost for a single branch"""

    def __init__(self, link_name, link_offsets, link_child_offsets, force_profiles: ForceProfile,
                 true_pos_trajectory, true_vel_trajectory, ignore_velocity_cost, grasp_locs):
        self.link_name = link_name
        self.link_offsets = link_offsets  # gym offset of the branch link
        self.link_child_offsets = link_child_offsets  # gym offset of the branch's child to estimate the center of mass
        self.force_profiles = force_profiles
        self.true_pos_trajectory = true_pos_trajectory  # true pos trajectory from ground truth
        self.true_vel_trajectory = true_vel_trajectory  # true vel trajectory from ground truth
        self.ignore_velocity_cost = ignore_velocity_cost
        self.grasp_locs = grasp_locs
        self.validate_traj()

    def validate_traj(self):
        num_steps, _, _ = ForceProfile.profile_props(self.force_profiles)
        assert num_steps == self.true_pos_trajectory.shape[0] == self.true_vel_trajectory.shape[0], "Invalid Traj Shape"
