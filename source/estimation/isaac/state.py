from isaacgym import gymapi
import numpy as np
import torch
from estimation.math import common


class GymState:
    def __init__(self, gym, sim, num_envs, num_bodies, dofs_per_tree_per_env, envs, logging):
        self.gym = gym
        self.sim = sim

        self.num_envs = num_envs  # total number of environments
        self.num_bodies = num_bodies  # total number of bodies across all environments

        self.dofs_per_tree_per_env = dofs_per_tree_per_env

        self.envs = envs
        self.logging = logging

    @staticmethod
    def _link_world_position(rb_positions, link_offsets):
        """ Returns the World position of the link origin"""
        return rb_positions[link_offsets, :]

    @staticmethod
    def validate_parent_child(link_offsets, link_child_offsets):
        """ Ensure parent child relationship b/w the offsets."""
        assert len(link_offsets) == len(link_child_offsets)
        for p, c in zip(link_offsets, link_child_offsets):
            assert p < c

    def link_world_position_at_dist(self, rb_positions, link_offsets, link_child_offsets, grasp_loc_t):
        """Find the coordinates of the grasp location at distance {grasp_loc} from the parent origin
        grasp_locs is calculated as (dist from origin)/(len of branch). grasp_locs is a list of size num_envs
        Default 0.5 for the mid-point (a.k.a center of mass), 0.25 for 2cm from the origin for a 8 cm branch, e.t.c."""

        self.validate_parent_child(link_offsets, link_child_offsets)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        link_origin_pos = self._link_world_position(rb_positions, link_offsets)
        link_child_origin_pos = self._link_world_position(rb_positions, link_child_offsets)
        return (link_origin_pos * (1.0 - grasp_loc_t)) + (grasp_loc_t * link_child_origin_pos)

    def link_length(self, tree_handle, rb_positions, link_offsets, link_child_offsets):
        """ Find the length of the link, given the link_offsets & link_child_offsets
            Find the link lengths using 2 methods and cross-check"""
        link_id = link_offsets[0]
        link_child_id = link_child_offsets[0]
        # 0 because all environments have the same link_lengths
        rigid_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], tree_handle)
        link_length = rigid_body_props[link_id].com.z * 2

        # this is just to validate that the length is correct
        rb_positions_view = rb_positions.view(self.num_envs, self.num_bodies, 3).detach().clone()

        all_env_link_lengths = common.l2_norm(rb_positions_view[:, link_id, :], rb_positions_view[:, link_child_id, :])
        assert abs(link_length - all_env_link_lengths[0].item()) < 0.1, "Invalid link lengths, check the child link pos"
        return link_length

    @staticmethod
    def _link_world_lin_vel(rb_linvels, link_offsets):
        return rb_linvels[link_offsets, :]

    def link_world_lin_vel_at_dist(self, rb_linvels, link_offsets, link_child_offsets, grasp_loc_t):
        """Find the vel of the grasp location at distance {grasp_loc} from the parent origin
        grasp_loc is calculated as (dist from origin)/(len of branch).
        Default 0.5 for the mid-point (a.k.a center of mass), 0.25 for 2cm from the origin for a 8 cm branch, e.t.c."""

        self.validate_parent_child(link_offsets, link_child_offsets)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        link_origin_lin_vel = self._link_world_lin_vel(rb_linvels, link_offsets)
        link_child_origin_lin_vel = self._link_world_lin_vel(rb_linvels, link_child_offsets)
        return (link_origin_lin_vel * (1.0 - grasp_loc_t)) + (grasp_loc_t * link_child_origin_lin_vel)

    @staticmethod
    def grasp_loc_t(grasp_locs):
        """ Get a tensor of grasp location in the shape of pos/vel tensor of the link
            In: list of size [num_env], Out:  tensor shape: num_env X 3 """
        all([0. < gl <= 1.0 for gl in
             grasp_locs]), "Invalid grasp point. Should be > 0.& <= 1.0, i.e. (dist frm origin)/(branch len)"
        grasp_locs_t = torch.tensor(grasp_locs)

        # use 3 to represent x, y, z dimensions
        grasp_loc_cart = grasp_locs_t.unsqueeze(1).repeat_interleave(3, dim=1)
        return grasp_loc_cart

    def get_link_offsets(self, tree_handle, link_name):
        """ Returns the offset of the same link in each environment
            tree_handle  refers to gt or sim
        """
        return [self.gym.find_actor_rigid_body_index(self.envs[e],
                                                     tree_handle,
                                                     link_name,
                                                     gymapi.DOMAIN_SIM)
                for e in range(self.num_envs)]

    def get_joint_offsets(self, tree_handle, dof_name):
        """ Returns the offset of the same dof in each environment
            tree_handle  refers to gt or sim
        """
        return [self.gym.find_actor_dof_index(self.envs[e],
                                              tree_handle,
                                              dof_name,
                                              gymapi.DOMAIN_SIM)
                for e in range(self.num_envs)]

    def get_parameters_by_env(self, env_handle, tree_handle):
        props = self.gym.get_actor_dof_properties(self.envs[env_handle], tree_handle)
        return props["stiffness"], props["damping"], props["friction"]

    def set_params_by_env(self, env_handle, tree_handle, p_stiffness_vals, p_damping_vals, p_friction=0.0):
        """ Set parameters for a single env and a single tree
            Input shape : np array (dofs_per_tree_per_env,)
            E.g. p_stiffness = np.full(shape=dofs_per_tree_per_env,fill_value=5000.,dtype=np.float32)
            Friction currently set to 0.0
        """
        props = self.gym.get_actor_dof_properties(self.envs[env_handle], tree_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"] = p_stiffness_vals
        props["damping"] = p_damping_vals
        props["friction"].fill(p_friction)
        self.gym.set_actor_dof_properties(self.envs[env_handle], tree_handle, props)

    def set_distr_tree_params(self, tree_handle, target_dofs, distr_p_stiffness, distr_p_damping,
                              default_p_stiffness_val, default_p_damping_val):
        """
            Given an array of parameters set it to the corresponding environment
            Each env in the tensor will have a separate stiffness attached t0 it.

            Input E.g:
                target_dofs = [dof_id1, dof_id2, ....]
                distr_p_stiffness = dict{
                    dof_id1  => [s1, s2, ....s-env]
                    dof_id2  => [s1, s2, ....s-env]
                }

            default_stiffness, default_damping: default values for all other dofs (except the targets).
            Warning: All d.o.f parameters s will be reset, but without actuation

        """
        self.logging.debug(f"target_dofs: {target_dofs}")
        self.logging.debug(f"distr_p_stiffness: {distr_p_stiffness}")
        self.logging.debug(f"distr_p_damping: {distr_p_damping}")

        if type(target_dofs) != list or len(target_dofs) <= 0:
            raise ValueError("Target DOFs should be a non-empty list", target_dofs)

        if not len(target_dofs) == len(distr_p_stiffness) == len(distr_p_damping):
            raise ValueError("Invalid shape for parameter dict")

        if len(distr_p_stiffness.get(target_dofs[0])) != self.num_envs:
            raise ValueError(f"Invalid stiffness parameter shape {len(distr_p_stiffness.get(target_dofs[0]))}")

        if len(distr_p_damping.get(target_dofs[0])) != self.num_envs:
            raise ValueError("Invalid damping parameter shape")

        for e in range(self.num_envs):
            # Default stiffness & damping for all branches of the tree
            sim_p_stiffness_vals = np.full(shape=self.dofs_per_tree_per_env,
                                           fill_value=default_p_stiffness_val,
                                           dtype=np.float32)
            sim_p_damping_vals = np.full(shape=self.dofs_per_tree_per_env,
                                         fill_value=default_p_damping_val,
                                         dtype=np.float32)

            # change specific parameter values
            for dof_id in target_dofs:
                distr_p_stiffness_by_dof = distr_p_stiffness[dof_id]
                sim_p_stiffness_vals[dof_id] = distr_p_stiffness_by_dof[e]

                distr_p_damping_by_dof = distr_p_damping[dof_id]
                sim_p_damping_vals[dof_id] = distr_p_damping_by_dof[e]

            self.set_params_by_env(e, tree_handle, sim_p_stiffness_vals, sim_p_damping_vals, p_friction=0.0)


class ArtifactState(object):
    """ A place-holder for all body/dof states to pass around"""

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ArtifactState, cls).__new__(cls)
            return cls.instance
        else:
            raise ValueError("Only one instance must be created for Artifact State class")

    def __init__(self, root_tensor, _root_tensor, dof_states, _dof_states, rb_states, _rb_states):
        # wrap it in a PyTorch Tensor and create convenient views
        self.root_tensor = root_tensor
        self.root_positions = self.root_tensor[:, 0:3]
        self.root_orientations = self.root_tensor[:, 3:7]
        self.root_linvels = self.root_tensor[:, 7:10]
        self.root_angvels = self.root_tensor[:, 10:13]

        self.dof_states = dof_states
        self._dof_states = _dof_states

        self.rb_states = rb_states
        self._rb_states = _rb_states

        self.rb_positions = self.rb_states[:, 0:3]
        self.rb_orientations = self.rb_states[:, 3:7]
        self.rb_linvels = self.rb_states[:, 7:10]
        self.rb_angvels = self.rb_states[:, 10:13]

        self.init_rb_positions = None
        self.init_rb_linvels = None

    def set_init_rb_states(self, init_rb_states):
        """ Set init values for pos/velocity after parameters have been set for the first time."""
        self.init_rb_positions = init_rb_states[:, 0:3]
        self.init_rb_linvels = init_rb_states[:, 7:10]
