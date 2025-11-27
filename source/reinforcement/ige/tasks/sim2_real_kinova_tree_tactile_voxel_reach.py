""" This task aims to reach a voxel with the tree branches acting as obstruction
    Additionally we use this class to capture and use joint torques/joint angles from the environment.
    This task can handle either torques based states or true-points based states
"""
from typing import Callable, Any

import numpy as np
import sys
import os
import torch
import pickle
import math
from estimation.real.domain import ArmType
from isaacgym import gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, tf_combine
from isaacgymenvs.tasks.base.vec_task import VecTask
import time
import joblib
from datetime import datetime

spot_path = os.getenv('spot_path')
source_dir = f"{spot_path}/source/"

_sim_set = 'set_k3'  # k for kinova
_real_set = 'set_p3'

# The directory to store forces, which can be inspected through a jupyter notebook.
_tactile_root_dir = f"{spot_path}/notebooks/work2/data/tactile/simulation/rl/collisions"
_tactile_raw_dir = f"{_tactile_root_dir}/{_sim_set}/raw"
# sequence of joint torques and contact forces indicator.
torque_cf_seq_file: Callable[[Any, Any], str] = lambda f_type, idx: f"{_tactile_raw_dir}/{f_type}_col_seq_{idx}.pt"
tactile_const_file = f"{_tactile_raw_dir}/tactile_constants.pkl"

# save the fixed robot_local_grasp_pos & robot_local_grasp_rot during train so that it can be used for real eval
robot_const_file = f"{_tactile_raw_dir}/robot_constants.pkl"

_tactile_test_dir = f"{_tactile_root_dir}/{_sim_set}/test"
_tactile_real_dir = f"{_tactile_root_dir}/{_sim_set}/real"

# can be executed only in sim
eval_voxel_fp = lambda eval_dir: f"{eval_dir}/in/voxel_pose_kinova_sim_multiple.txt"

# can be executed in both sim and real
# eval_voxel_fp = lambda eval_dir: f"{eval_dir}/in/voxel_pose_kinova_real_multiple.txt"

# by changing the rotation/degrees of the tree with the same policy, multiple test_patterns can be evaluated.
test_tree_rot = 315.0  # 285.0, 300.0, 310.0, 315.0 (default),  #320.0
_trot = f"rot{int(test_tree_rot)}"
_test_run_seed = 0
eval_dof_torques_abs_sum_fp = (lambda eval_dir, chk:
                               f"{eval_dir}/out/dof_torques_abs_sum_by_frame_{_trot}_run{_test_run_seed}_{chk}_time.pkl")
eval_net_cf_abs_sum_fp = (lambda eval_dir, chk:
                          f"{eval_dir}/out/net_cf_abs_sum_by_frame_{_trot}_run{_test_run_seed}_{chk}_time.pkl")
eval_succ_tracker_fp = (lambda eval_dir, chk:
                        f"{eval_dir}/out/succ_tracker_{_trot}_run{_test_run_seed}_{chk}_time.pkl")

real_coll_path = f"{spot_path}/notebooks/work2/data/tactile/external/kinova/rl/collisions/{_real_set}"
trained_classifier_path = f"{real_coll_path}/classifier/model/rf/classifier_latest.pkl"  # RF
# trained_classifier_path = f"{real_coll_path}/classifier/model/nn/classifier_latest.pth" # NN

if source_dir not in sys.path:
    sys.path.append(source_dir)

from common.helpers import futils
from reinforcement.ige.helpers.rl_helper import arm_random_loc_within_reach
from reinforcement.ige.helpers import physics_helper, rl_helper
from reinforcement.ige.real.kinova import robot_interface as rki
from reinforcement.ige.real.kinova import robot_domain
from reinforcement.ige.real.kinova.robot_collision_checker import store_frame_lag
from reinforcement.ige.helpers.domains import SuccTracker, CollisionPenalty
from reinforcement.ige.real.kinova.classifiers import nnc as nnc
from reinforcement.ige.real.kinova.classifiers import rfc as rfc

cmd_args = {}
for arg in sys.argv[1:]:
    key, value = arg.split("=")
    cmd_args[key] = value

init_steps_to_skip = 10
# if true-points are used for RL, choose a subset of branches for passing as the state.
num_branches_to_select = 20
dof_torque_obs_scale = 1.

store_frame_freq = 100  # store trg torques every x frames


# To run:  bash $spot_path/source/reinforcement/ige/ige_task_runner.sh task=Sim2RealKinovaTreeTactileVoxelReach
# Optional Arguments: test=True num_envs=64 headless=True +real=True
# nomenclature : eval can be for real kinova (called self.real in code) or eval in simulation (called self.test in code)
# nomenclature : For eval in real kinova (args test=True +real=True)  for eval in simulation (args test=True)
# noinspection PyAttributeOutsideInit
class Sim2RealKinovaTreeTactileVoxelReach(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # is the policy already trained, and is running with test=True argument.
        self._arg_test = bool(cmd_args.get('test', False))

        # is the policy already trained, and is running with real=True (test in real world) argument.
        self._arg_real = bool(cmd_args.get('+real', False))
        self._arg_headless = bool(cmd_args.get('headless', True))

        # monitor saves values of ja,jv,jt to compare sim vs real. Will impact performance, keep off except for test.
        self._arg_monitor_metric = bool(cmd_args.get('+monitor_metric', False))

        if self._arg_real and not self._arg_test:
            raise ValueError("Invalid combination: To enable evaluation in real, test must be set to true.")
        else:
            self.real = self._arg_real
            self.test = self._arg_test and not self._arg_real

        print(f"Train/Test/Real => Test:{self.test}, Real:{self.real}")

        # the below two are valid only if self.real = True, and for a single env
        # it keeps track of actions & collisions attempted in th real world
        self.action_attempts = 0
        self.real_collision_count = 0
        self.real_no_collision_count = 0
        self.real_coll_prob_thresh = 0.4  # threshold beyond which an interaction is classified as collision.

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.collision_reward_scale = self.cfg["env"]["collisionRewardScale"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.dt = 1 / 60
        self.kinova_cont_dof_ind = [0, 3, 4, 5]  # indices of dofs with continuous joints,

        # changed observation space to include a few branch locations as well
        num_obs = 20  # ja (6) + jv (6) + ee-tp (3) + classifier(1) + hand_rot (4)
        num_acts = 6

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # applicable only to train, RL training is independent of torque capture.
        self.max_trg_torques_to_store = self.cfg["env"]["maxTrgTorquesToStore"]
        self.store_trg_torques = self.cfg["env"]["storeTrgTorques"] and self.max_trg_torques_to_store > 0
        self.enable_joint_torque_obs = self.cfg["env"]["enableJointTorqueObs"]
        self.collision_penalty_type = CollisionPenalty.ptype(self.cfg["env"]["collisionPenalty"])

        # use voxels from file as test targets. applicable only for inference
        self.enable_eval_voxel_file = self.cfg["env"]["enableEvalVoxelFile"]
        self.brush_past_norm_cf = self.cfg["env"]["brushPastNormContactForce"]
        self.transferable_to_real = self.cfg["env"]["transferableToReal"]
        self.dynamics_by_beam_deflection = self.cfg["env"]["dynamicsByBeamDeflection"]
        self.randomise_robot_start_pose = self.cfg["env"]["randomiseRobotStartPose"]
        self.enable_symmetry_awareness = self.cfg["env"]["enableSymmetryAwareness"]
        self.test_tree_use_file_rotation = self.cfg["env"]["testTreeUseFileRotation"]

        # useful only to store trainable torques
        self.stored_torque_set_count = 0  # stored so far.

        #  number of resets of environments due to instability in environments
        # This 3 lines are added as a fix for the robot nan return problem
        self.sim_instability_resets = 0

        #  I think the environment, assets, actors all get created on the super call using the
        # over-loaded methods below _create_envs, _create_sim, e.t.c.
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self.validate_settings()

        # Note: to execute home pose the clamp limits for continuous joints should be at least -360d to +360d
        # Kinova home pose : ([4.661, 3.493, 1.623, 4.211, 0.909, 1.544]
        # Kinova mean (avg(up,lo)) pose : [0.0000, 3.1416, 3.1416, 0.0000, 0.0000, 0.0000]
        # Kinova upright pose facing the tree: [0.0000, 3.1416, 3.1416, 1.78, 3.1416, 0.0000]
        self.kinova_default_dof_pos = to_torch([4.661, 3.493, 1.623, 4.211, 0.909, 1.544],
                                               device=self.device)

        self.eval_dir = _tactile_real_dir if self.real else _tactile_test_dir

        if self.test or self.real:
            self.eval_succ_tracker = SuccTracker()  # number of succ/fails
            self.eval_voxel_pos_gen = rl_helper \
                .tensors_from_file(file_path=eval_voxel_fp(self.eval_dir),
                                   device=self.device) if self.enable_eval_voxel_file else None

        if self.real:
            self.init_real_kinova_attrs()  # invoke in case of test with real arm
        else:
            self.init_sim_attrs()  # invoke in case of train & test in sim

        # self.loaded_classifier_model = torch.load(trained_classifier_path).to(self.device) # for NN
        self.loaded_classifier_model = joblib.load(trained_classifier_path)  # for RF

        if self._arg_monitor_metric:
            assert (self.test or self.real), "no monitors during training."
            run_type = robot_domain.RunType.REAL if self.real else robot_domain.RunType.SIM
            checkpoint_dir = rl_helper.extract_checkpoint_dir(cmd_args['checkpoint'])
            self.jm_monitor = robot_domain.JointMetricMonitor(
                device=self.device,
                check_point=checkpoint_dir,
                num_robot_dofs=self.num_robot_dofs,
                run_type=run_type)

    def init_real_kinova_attrs(self):

        assert self.num_envs == 1, "In real only one env is allowed"

        # only one of the below will be used either pos control or vel control
        # in real & sim no_dofs are different. In real only robot is included.
        self.robot_dof_pos_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.robot_dof_vel_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def init_sim_attrs(self):

        # get gym GPU state tensors

        # All root bodies' position, orientation, linear velocity, and angular velocity of actors.
        # The shape of this tensor is (envs, num_actors, 13) a
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        # Dof of actors. The shape of the tensor is (envs, num_dofs, 2).  2= pos in radians, vel/rad per seco
        # Sequential placement . Begins with all DOFs of actor 0, followed by all the DOFs of actor 1, & so on.
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        # All rigid body position, orientation, linear velocity, and angular velocity
        # The shape is (envs, num_rigid_bodies, 13) with sequential placement.
        _rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # panda has 11 links and 9 joints.
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.robot_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_robot_dofs]
        self.robot_dof_pos = self.robot_dof_state[..., 0]
        self.robot_dof_vel = self.robot_dof_state[..., 1]

        # take out the tree state from the global dof state.
        self.tree_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_robot_dofs:]

        self.tree_dof_pos = self.tree_dof_state[..., 0]
        self.tree_dof_vel = self.tree_dof_state[..., 1]

        rigid_body_tensor = gymtorch.wrap_tensor(_rigid_body_tensor)
        self.rigid_body_states = rigid_body_tensor.view(self.num_envs, -1, 13)
        self.branch_poses = self.rigid_body_states[:, (self.num_robot_bodies + 1):, 0:3]

        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        # num_dofs per env
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        # only one of the below will be used either pos control or vel control
        self.robot_dof_pos_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.robot_dof_vel_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        _dof_torques = self.gym.acquire_dof_force_tensor(self.sim)
        dof_torques = gymtorch.wrap_tensor(_dof_torques).view(self.num_envs, self.num_dofs)

        # Assuming that robot is the first body
        self.robot_dof_torques = dof_torques[:, :self.num_robot_dofs]

        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, self.num_bodies, 3)

        # we use forces at t0, t-1, t-2 to smoothen the noise;code could be better.
        self.curr_robot_net_cf = net_cf[:, :self.num_robot_bodies, :]
        self.prev0_robot_net_cf = torch.zeros_like(self.curr_robot_net_cf)
        self.prev1_robot_net_cf = torch.zeros_like(self.curr_robot_net_cf)
        self.prev2_robot_net_cf = torch.zeros_like(self.curr_robot_net_cf)

        # this variable is used to reset the forces in refresh tensor when a reset happens
        self.recent_reset_env_ids_frc = torch.empty(0, dtype=torch.int64).to(self.device)

        print("robot dof torque shape", self.robot_dof_torques.shape)
        print("current robot net_cf shape", self.curr_robot_net_cf.shape)

        # This additional refresh_all_sim_tensors lines are added as per https://forums.developer.nvidia.com/t
        # /isaacgym-preview-4-actor-root-state-returns-nans-with-isaacgymenvs-style-task/223738/3
        self.refresh_all_sim_tensors()

        for i in range(init_steps_to_skip):
            print(f"Skipping steps ....{i} to allow the branches to settle and get the default state of the tree..")
            self.render()
            self.gym.simulate(self.sim)
            self.refresh_all_sim_tensors()

        # take a backup to use during resets.
        self.tree_default_dof_state = self.tree_dof_state.detach().clone()

        # capture contact force/succ  metric to store during tests.
        if self.test:
            self.dof_torques_abs_sum_by_frame = []  # list of lists. Each test goes into a list.
            self.net_cf_abs_sum_by_frame = []
            # number of rl steps (prog_buf) taken to reach target (touch voxel)
            # if reach fails, steps = maximum episode length.
            self.test_steps_to_succ = self.max_episode_length

        # env-actor global index 2(robot/tree) + 1 (voxel)
        self.global_indices = torch.arange(self.num_envs * (2 + 1), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        # env global index
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def validate_settings(self):

        if self.real:
            assert self._arg_headless is True, "In real turnoff graphics, else the frequencies will mess up"
            assert self.transferable_to_real is True, "This policy was not trained to be transferable."

        if self.test or self.real:
            assert (self.test and not self.real) or (self.real and not self.test), "Only one (test or real) should be T"
            assert self.num_envs <= 64, "To avoid rendering overhead, run with fewer environments using num_envs=64"
            # assert self.headless is False, "Render the UI to observe result during test."
            checkpoint_dir = rl_helper.extract_checkpoint_dir(cmd_args['checkpoint'])
            rl_helper.match_train_test_confs(test_conf=self.cfg,
                                             train_yaml_path=f"{spot_path}/runs/{checkpoint_dir}/config.yaml")
        if self.test and 'ablation' not in self.lsystem_asset_root:
            assert self.test_tree_use_file_rotation is False, "Use learned rotation only during l-system ablation"

        if (self.test or self.real) and self.enable_eval_voxel_file:
            assert self.num_envs == 1, "pre-specified target can be used only with 1 environment"

        if self.test:
            assert self.num_envs == 1, "test_steps_to_succ assumes only 1 env during sim test"

        assert not (self.store_trg_torques
                    and self.enable_joint_torque_obs), "Either store trg torques or enable tactile inp for predictions"
        assert not (self.store_trg_torques and self.test), "Can't store trg torques during inference."
        assert not (self.store_trg_torques and self.real), "Can't store trg torques during inference."

        if self.store_trg_torques:
            assert self.num_envs == 512, "Currently to store training torques, fix the num_envs at 512 "
            assert store_frame_lag < store_frame_freq, "Invalid combinations for frequency and lag."
            print("===== Warning:Joint Torques/ Collision information are being stored/replaced locally. =====")

        if not self.real:
            all_num_branches = [self.gym.get_asset_rigid_body_count(ta) for ta in self.tree_assets]
            all_num_tree_dofs = [self.gym.get_asset_dof_count(ta) for ta in self.tree_assets]

            assert all(element == all_num_branches[0] for element in all_num_branches), "Not all tree body counts equal"
            assert all(element == all_num_tree_dofs[0] for element in all_num_tree_dofs), "Not all dof counts equal"

    def create_sim(self):  # The over-loaded function
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        #  call functions below
        self._create_ground_plane()

        tree_start_position = gymapi.Vec3(0.0, 0.0, 0.0)

        if self.real:
            self._create_real_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        else:
            self._create_sim_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)),
                                  tree_start_position)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_real_envs(self, spacing, num_per_row):
        """This is a dummy simulation that we use to execute in parallel to real. Need to be removed sometime.
           The values from this sim are not used. Just the envs[] list is required for IsaacGym Envs."""

        self.voxel_size = 0.05
        real_robot_start_position = gymapi.Vec3(1.0, 0.0, 0.0)
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        assert "asset" in self.cfg["env"]

        # load robot asset (franka/kinova)
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL  # switch to velocity control
        asset_options.use_mesh_materials = True

        self.robot_start_pose = gymapi.Transform()
        self.robot_start_pose.p = real_robot_start_position
        self.robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 0.0)
        # this is place-holder, "no" contact force in case of real.
        self.robot_net_cf = torch.empty(self.num_envs, 3)

        self.robots = []  # all robot handles
        self.envs = []

        # actor creation step per environment.
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            self.envs.append(env_ptr)

        self.init_real_data()

    def _create_sim_envs(self, num_envs, spacing, num_per_row, tree_start_position):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        assert "asset" in self.cfg["env"]
        common_asset_root = os.path.join(f"{spot_path}", self.cfg["env"]["asset"].get("commonAssetRoot"))
        kinova_asset_file = self.cfg["env"]["asset"].get("assetFileNameKinova")

        train_lsystem_asset_root = os.path.join(f"{spot_path}", self.cfg["env"]["asset"].get("trainLsystemAssetRoot"))
        test_lsystem_asset_root = os.path.join(f"{spot_path}", self.cfg["env"]["asset"].get("testLsystemAssetRoot"))
        lsystem_asset_root = test_lsystem_asset_root if self.test else train_lsystem_asset_root
        self.lsystem_asset_root = lsystem_asset_root

        self.asset_rel_path = self.cfg["env"]["asset"].get("assetFileNameTree")
        tree_alignment_quats_file = os.path.join(lsystem_asset_root, 'tree_alignment_quats.pkl')

        if os.path.exists(tree_alignment_quats_file):
            with open(tree_alignment_quats_file, 'rb') as file:
                tree_alignment_quats = pickle.load(file)

        else:
            raise ValueError("Run tree_base_rotation_calculator.py to compute the optimal rotation first.")

        self.num_tree_types = futils.count_sub_folders(lsystem_asset_root)
        # repeat excluding 0 like [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        repeated_file_indices = [j % self.num_tree_types for j in range(self.num_envs)]
        assert len(tree_alignment_quats) == self.num_tree_types, "invalid alignment information"
        rep_tree_alignment_quats = [tree_alignment_quats[rep_idx] for rep_idx in repeated_file_indices]

        assert self.num_envs >= self.num_tree_types >= 1, "Invalid value: num_tree_types"

        tree_asset_files = [self.asset_rel_path.format(r_idx=r_idx) for r_idx in repeated_file_indices]

        if self.dynamics_by_beam_deflection:
            tree_meta_files = [asset_file.replace('.urdf', '_meta.pkl') for asset_file in tree_asset_files]

        # load robot asset (franka/kinova)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL  # switch to velocity control
        asset_options.use_mesh_materials = True
        # robot should be the first asset to be loaded. Changing the order will f*** up the indices.
        kinova_asset = self.gym.load_asset(self.sim, common_asset_root, kinova_asset_file, asset_options)

        # load tree asset
        tree_asset_options = gymapi.AssetOptions()
        tree_asset_options.fix_base_link = True
        tree_asset_options.armature = 0.01
        tree_asset_options.collapse_fixed_joints = True  # important for the tree.
        tree_asset_options.disable_gravity = False
        tree_asset_options.override_com = True
        tree_asset_options.override_inertia = True
        tree_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        self.tree_assets = []
        self.tree_metas = []
        for i in range(0, num_envs):
            _tree_asset = self.gym.load_asset(self.sim, lsystem_asset_root, tree_asset_files[i], tree_asset_options)
            self.tree_assets.append(_tree_asset)
            if self.dynamics_by_beam_deflection:
                with open(os.path.join(lsystem_asset_root, tree_meta_files[i]), 'rb') as file:
                    # Load the data from the pickle file
                    self.tree_metas.append(pickle.load(file))

        # Non-zero stiffness for DOF_MODE_POS
        # kinova_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400], dtype=torch.float,  device=self.device)
        # 0 stiffness for DOF_MODE_VEL
        kinova_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0], dtype=torch.float, device=self.device)

        kinova_dof_damping = to_torch([80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)
        # kinova_dof_damping = to_torch([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float, device=self.device)
        # kinova_dof_damping = to_torch([2., 2., 2., 2., 2., 2.], dtype=torch.float, device=self.device)
        # kinova_dof_damping = to_torch([200., 200., 200., 200., 200., 200.], dtype=torch.float, device=self.device)

        kinova_dof_friction = to_torch([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2], dtype=torch.float,
                                       device=self.device)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(kinova_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(kinova_asset)
        self.num_branches = self.gym.get_asset_rigid_body_count(self.tree_assets[0])
        self.num_tree_dofs = self.gym.get_asset_dof_count(self.tree_assets[0])

        print("num robot bodies: ", self.num_robot_bodies)
        print("num robot dofs: ", self.num_robot_dofs)
        print("num branches (rig bodies) in a tree: ", self.num_branches)
        print("num tree dofs: ", self.num_tree_dofs)
        print("store trg torques: ", self.store_trg_torques)
        print("use joint torque observations for state ? (vs use true points): ", self.enable_joint_torque_obs)
        print("penalise collision during reward: ", self.collision_penalty_type)
        print("DR Tree Types: ", self.num_tree_types)

        # set robot dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(kinova_asset)
        self.robot_dof_pos_lower_limits = []
        self.robot_dof_pos_upper_limits = []
        self.robot_dof_vel_max_limits = []
        self.robot_dof_effort_max_limits = []
        for i in range(self.num_robot_dofs):
            # DOF_MODE_POS/DOF_MODE_VEL is used for position/velocity control
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL  # switch to velocity control
            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_props['stiffness'][i] = kinova_dof_stiffness[i]
                robot_dof_props['damping'][i] = kinova_dof_damping[i]
                robot_dof_props['friction'][i] = kinova_dof_friction[i]
            else:
                raise ValueError("Not yet Implemented.")

            self.robot_dof_pos_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_pos_upper_limits.append(robot_dof_props['upper'][i])
            self.robot_dof_vel_max_limits.append(robot_dof_props['velocity'][i])
            self.robot_dof_effort_max_limits.append(robot_dof_props['effort'][i])

        print(f"Robot dof effort : {robot_dof_props['effort']}")
        print(f"Robot dof stiffness : {robot_dof_props['stiffness']}")
        print(f"Robot dof damping : {robot_dof_props['damping']}")
        print(f"Robot dof hasLimits : {robot_dof_props['hasLimits']}")
        print(f"Robot dof velocity : {robot_dof_props['velocity']}")
        print(f"Robot dof friction : {robot_dof_props['friction']}")

        self.robot_dof_pos_lower_limits = to_torch(self.robot_dof_pos_lower_limits, device=self.device)
        self.robot_dof_pos_upper_limits = to_torch(self.robot_dof_pos_upper_limits, device=self.device)
        self.robot_dof_vel_max_limits = to_torch(self.robot_dof_vel_max_limits, device=self.device)
        self.robot_dof_effort_max_limits = to_torch(self.robot_dof_effort_max_limits, device=self.device)

        # for continuous joints in Kinova the limit values are too high -3.4028e+38 to 3.4028e+38, preventing ops
        # so make it smaller to reduce freedom. 360d = one full rotation to either side.
        self.robot_dof_pos_lower_limits[self.kinova_cont_dof_ind] = -1 * math.radians(360)
        self.robot_dof_pos_upper_limits[self.kinova_cont_dof_ind] = math.radians(360)

        print(f"Robot dof lower limits : {self.robot_dof_pos_lower_limits}")
        print(f"Robot dof upper limits : {self.robot_dof_pos_upper_limits}")
        print(f"Robot dof mean limits : {(self.robot_dof_pos_lower_limits + self.robot_dof_pos_upper_limits) / 2}")

        print(f"Robot dof velocity limits : {self.robot_dof_vel_max_limits}")
        print(f"Robot dof effort limits : {self.robot_dof_effort_max_limits}")

        print(f"randomiseRobotStartPose ? : {self.randomise_robot_start_pose}")
        print(f"enableSymmetryAwareness ? : {self.enable_symmetry_awareness}")

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_pos_lower_limits)

        # create voxel assets, just an empty space indicator
        voxel_opts = gymapi.AssetOptions()
        voxel_opts.fix_base_link = True
        self.voxel_size = 0.05
        voxel_color = gymapi.Vec3(1.0, 0., 0.)
        # voxel_asset = self.gym.create_box(self.sim, voxel_size, voxel_size, voxel_size, voxel_opts)
        voxel_asset = self.gym.create_sphere(self.sim, self.voxel_size, voxel_opts)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(kinova_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(kinova_asset)

        num_tree_bodies = self.gym.get_asset_rigid_body_count(self.tree_assets[0])
        num_tree_shapes = self.gym.get_asset_rigid_shape_count(self.tree_assets[0])

        num_voxel_bodies = self.gym.get_asset_rigid_body_count(voxel_asset)
        num_voxel_shapes = self.gym.get_asset_rigid_shape_count(voxel_asset)

        max_agg_bodies = num_robot_bodies + num_tree_bodies + num_voxel_bodies
        max_agg_shapes = num_robot_shapes + num_tree_shapes + num_voxel_shapes

        self.robots = []  # all robot handles
        self.trees = []  # all tree handles
        self.voxels = []
        self.envs = []

        # actor creation step per environment.
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            # An aggregate is a collection of actors. This is for performance improvement.
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self.robot_start_pose = gymapi.Transform()
            sim_robot_start_xyz_default = gymapi.Vec3(1.0, 0.0, 0.0)
            sim_robot_start_rot_default = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

            if self.randomise_robot_start_pose:
                _seed = _test_run_seed if self.test else i  # repeatable for training, but random during test.
                # random_x = rl_helper.rand_position_single_axis(low=0.75, high=1.0, seed=_seed)
                # self.robot_start_pose.p = gymapi.Vec3(random_x, 0.0, 0.0)

                self.robot_start_pose.p = sim_robot_start_xyz_default
                self.robot_start_pose.r = rl_helper.rand_rotate(original_quat=sim_robot_start_rot_default,
                                                                rand_rot_limit_degree=180, seed=_seed)
            else:
                # The rotation of the trees depend on robot position, so should be fixed at (1.0, 0.0, 0.0)
                self.robot_start_pose.p = sim_robot_start_xyz_default
                self.robot_start_pose.r = sim_robot_start_rot_default

            # It is common to have one collision group per environment, in which case the group id 'i'
            # corresponds to environment index. This prevents actors in different environments from collision.
            # The 0/1/2 after i is the collision_filter. 0 = no filtering & enable self collisions.
            # Two bodies will not collide if they are in diff group or if their filters have a common bit set
            # So in the below case the robot and the tree will collide, but neither will collide with voxel.
            # enable self collisions in the robot with the final 0.

            robot_actor = self.gym.create_actor(env_ptr, kinova_asset, self.robot_start_pose, "kinova", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            tree_start_pose = gymapi.Transform()
            tree_start_pose.p = tree_start_position

            # make slight variation to the rot placement of the tree for Domain Randomisation
            rand_quat = rl_helper.rand_rotate(original_quat=rep_tree_alignment_quats[i], rand_rot_limit_degree=5)
            tree_start_pose.r = rand_quat

            # move around z-axis, the robot and the tree can collide depending on the angle.
            if self.test:
                if self.test_tree_use_file_rotation:
                    print("Warning: Using tree rotation from file:", rep_tree_alignment_quats[i])
                    tree_start_pose.r = rep_tree_alignment_quats[i]
                else:
                    print("Using fixed tree test_tree_rot :", test_tree_rot)
                    tree_start_pose.r = gymapi.Quat.from_axis_angle(
                        gymapi.Vec3(0, 0, 1.0), np.radians(test_tree_rot))

            tree_pose = tree_start_pose

            tree_actor = self.gym.create_actor(env_ptr, self.tree_assets[i], tree_pose, "tree", i, 1)

            # the scaling upsets the c.o.m, the forces e.t.c

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # randomly generate a location to place the target voxel
            voxel_random_pose = arm_random_loc_within_reach(arm_base_pose=self.robot_start_pose,
                                                            arm_type=ArmType.kinova)

            # use a different collision group for voxel to avoid interaction b/w voxel & (tree/robot)
            voxel_actor = self.gym.create_actor(env_ptr, voxel_asset, voxel_random_pose,
                                                "fixed_voxel", self.num_envs + i, 0)
            self.gym.set_rigid_body_color(env_ptr, voxel_actor, 0, gymapi.MESH_VISUAL, voxel_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.trees.append(tree_actor)
            self.voxels.append(voxel_actor)

            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_actor)

        # set tree dof properties
        for env_i in range(num_envs):
            tree_dof_props = self.gym.get_asset_dof_properties(self.tree_assets[env_i])
            for dof_i in range(self.num_tree_dofs):
                noise_std = 0. if self.test else 1.  # Noise only for train for DR.
                if self.dynamics_by_beam_deflection:
                    child_br_name = rl_helper.child_branch_name(
                        self.gym.get_asset_dof_name(self.tree_assets[env_i], dof_i))
                    child_br_shape = self.tree_metas[env_i].l_shape_dict[child_br_name]
                    kp, kd = physics_helper.beam_deflection_param(
                        radius=child_br_shape.radius, length=child_br_shape.length, noise_std=noise_std)
                    tree_dof_props['stiffness'][dof_i] = kp
                    tree_dof_props['damping'][dof_i] = kd
                    tree_dof_props['friction'][dof_i] = 0.01  # some low value
                else:
                    # use rudimentary policy for test and train
                    branch_level = rl_helper.branch_level(self.gym.get_asset_dof_name(self.tree_assets[env_i], dof_i))
                    kp, kd = physics_helper.rud_deflection_param(branch_level=branch_level, noise_std=noise_std)
                    tree_dof_props['stiffness'][dof_i] = kp
                    tree_dof_props['damping'][dof_i] = kd
                    tree_dof_props['friction'][dof_i] = 0.01  # some low value

            self.gym.set_actor_dof_properties(self.envs[env_i], self.trees[env_i], tree_dof_props)

            # some tree info
            if env_i < 5:
                print(f"Tree {env_i} stiffness:", sorted(list({*tree_dof_props['stiffness']})))
                print(f"Tree {env_i} damping:", sorted(list({*tree_dof_props['damping']})))
                print(f"Tree {env_i} friction:", sorted(list({*tree_dof_props['friction']})))

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, "j2n6s300_end_effector")

        # j2n6s300_link_finger_1 is the thumb
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, "j2n6s300_link_finger_1")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, robot_actor, "j2n6s300_link_finger_3")
        self.voxel_handle = self.gym.find_actor_rigid_body_handle(env_ptr, voxel_actor, "box")
        self.tree_root_handle = self.gym.find_actor_rigid_body_handle(env_ptr, tree_actor, "world")

        self.random_branch_indices = torch.randperm(self.num_branches)[:num_branches_to_select].to(self.device)

        if self.store_trg_torques:
            futils.cleanup_location(_tactile_raw_dir, file_types=(".png", ".jpg", ".npy", "pt", "pkl"))

        self.init_sim_data()

    def refresh_all_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # smoothen contact forces
        # t = (t0 + t-1 + t-2)/3;   t-2 = t-1;   t-1 = t0 ;  t0 = refresh

        self.prev2_robot_net_cf = self.prev1_robot_net_cf.detach().clone()
        self.prev1_robot_net_cf = self.prev0_robot_net_cf.detach().clone()

        # refresh forces
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.prev0_robot_net_cf = self.curr_robot_net_cf.detach().clone()

        # for the environments just reset, curr_robot_net_cf = 0 & and all prev net_cf 0
        self.prev0_robot_net_cf[self.recent_reset_env_ids_frc, :, :] = 0

        self.robot_net_cf = (self.prev0_robot_net_cf + self.prev1_robot_net_cf + self.prev2_robot_net_cf) / 3

        self.recent_reset_env_ids_frc = torch.empty(0, dtype=torch.int64).to(self.device)

    def init_real_data(self):

        assert self.real is True, "not supposed to be here"
        self.num_bodies = 16  # 3 x 2 for fingers + 6 links.
        self.num_dofs = 6
        self.num_robot_dofs = 6

        # store at least last n_lag records incoming from real arm
        self.curr_dof_torque_mem = torch.empty(0, self.num_dofs).to(self.device)

        with open(robot_const_file, 'rb') as file:
            robot_const = pickle.load(file)

        self.robot_local_grasp_pos = robot_const['robot_local_grasp_pos'].unsqueeze(0).to(self.device)
        self.robot_local_grasp_rot = robot_const['robot_local_grasp_rot'].unsqueeze(0).to(self.device)

        self.robot_dof_pos_lower_limits = robot_const['robot_dof_pos_lower_limits']
        self.robot_dof_pos_upper_limits = robot_const['robot_dof_pos_upper_limits']
        self.robot_dof_vel_max_limits = robot_const['robot_dof_vel_max_limits']
        self.robot_dof_effort_max_limits = robot_const['robot_dof_effort_max_limits']

        assert self.robot_local_grasp_pos.shape == torch.Size([self.num_envs, 3])
        assert self.robot_local_grasp_rot.shape == torch.Size([self.num_envs, 4])

        print("robot_local_grasp_pos:", self.robot_local_grasp_pos)
        print("robot_local_grasp_rot:", self.robot_local_grasp_rot)
        print("robot_dof_pos_lower_limits:", self.robot_dof_pos_lower_limits)
        print("robot_dof_pos_upper_limits:", self.robot_dof_pos_upper_limits)
        print("robot_dof_vel_max_limits:", self.robot_dof_vel_max_limits)
        print("robot_dof_effort_max_limits:", self.robot_dof_effort_max_limits)

        # initialise with 0s for all environments.
        self.robot_grasp_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_grasp_rot = torch.zeros_like(self.robot_local_grasp_rot)

        assert self.robot_local_grasp_pos.shape == torch.Size([1, 3])
        assert self.robot_local_grasp_rot.shape == torch.Size([1, 4])

        assert self.robot_dof_pos_lower_limits.shape == torch.Size([self.num_robot_dofs])
        assert self.robot_dof_pos_upper_limits.shape == torch.Size([self.num_robot_dofs])
        assert self.robot_dof_vel_max_limits.shape == torch.Size([self.num_robot_dofs])
        assert self.robot_dof_effort_max_limits.shape == torch.Size([self.num_robot_dofs])

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_pos_lower_limits)

    def init_sim_data(self):

        # j2n6s300_link_finger_1 is the thumb
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "j2n6s300_end_effector")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "j2n6s300_link_finger_1")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robots[0], "j2n6s300_link_finger_3")

        # get_rigid_transform: Vectorized bindings to get rigid body transforms in the env frame.
        # different from rigid body states.
        # The transform poses are the CURRENT in the environments coordinate system
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        # The grasp point is the mid-point of the fingers.
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5

        # # assuming fingers are parallel.
        finger_pose.r = lfinger_pose.r

        # The poses are in the environment's coordinate system. However, they are also a gymapi transform
        # A transform here gives the pose w.r.t to env coordinate frame.
        # In the initial pose the fingers are pointing down
        print(f"hand_pose.p: {hand_pose.p}")  # hand_pose.p: Vec3(0.885231, -0.000000, 1.031682)
        print(f"finger_pose.p: {finger_pose.p}")  # finger_pose.p: Vec3(0.896767, 0.000000, 0.866684)

        # We convert the position of the grasp point in the hand coordinate frame.
        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        # this provides the location of the fingers in the hand coordinate frame.
        # robot_local_grasp_pose is the current location of the
        #   finger gasp point w.r.t the c.o.m of the hand  (panda_link7)
        robot_local_grasp_pose = hand_pose_inv * finger_pose

        # 1. franka_local_grasp_pose.p: Vec3(0.000000, 0.000000, 0.165400)
        # approximately 1.031682 - 0.866684, seems the hand and fingers are aligned initially.
        # Essentially means the finger grasp point is 16cm in the z-axis from the hand
        print(f"1. robot_local_grasp_pose.p: {robot_local_grasp_pose.p}")

        # Some doco suggests The maximum width of the open gripper is 80 mm (8cm)
        # The 0.04 or 4cm is probably the thickness of the hand which is in the local y-axis
        robot_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))

        # 2. franka_local_grasp_pose.p: Vec3(0.000000, 0.040000, 0.165400)
        print(f"2. robot_local_grasp_pose.p: {robot_local_grasp_pose.p}")

        # The repeat expands the tensor to all envs.
        # The robot_local_grasp_rot, robot_local_grasp_pos are fixed throughout, doesn't change during exec
        self.robot_local_grasp_pos = to_torch([robot_local_grasp_pose.p.x, robot_local_grasp_pose.p.y,
                                               robot_local_grasp_pose.p.z], device=self.device).repeat(
            (self.num_envs, 1))
        self.robot_local_grasp_rot = to_torch([robot_local_grasp_pose.r.x, robot_local_grasp_pose.r.y,
                                               robot_local_grasp_pose.r.z, robot_local_grasp_pose.r.w],
                                              device=self.device).repeat((self.num_envs, 1))

        if not self.test and not self.real:
            # store the fixed local grasps for a single environment
            print("self.robot_local_grasp_pos:", self.robot_local_grasp_pos)
            print("self.robot_local_grasp_rot:", self.robot_local_grasp_rot)

            robot_const = {
                'robot_local_grasp_pos': self.robot_local_grasp_pos[0],
                'robot_local_grasp_rot': self.robot_local_grasp_rot[0],
                'robot_dof_vel_max_limits': self.robot_dof_vel_max_limits,
                'robot_dof_pos_lower_limits': self.robot_dof_pos_lower_limits,
                'robot_dof_pos_upper_limits': self.robot_dof_pos_upper_limits,
                'robot_dof_effort_max_limits': self.robot_dof_effort_max_limits
            }

            with open(robot_const_file, 'wb') as file:
                pickle.dump(robot_const, file)

        # I think this is the gripper (hand) local coordinate system axis
        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        # initialise with 0s for all environments.
        self.robot_grasp_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_grasp_rot = torch.zeros_like(self.robot_local_grasp_rot)

        # set the quartenion last element to be 1.
        self.robot_grasp_rot[..., -1] = 1  # xyzw

        self.robot_lfinger_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_rfinger_pos = torch.zeros_like(self.robot_local_grasp_pos)
        self.robot_lfinger_rot = torch.zeros_like(self.robot_local_grasp_rot)
        self.robot_rfinger_rot = torch.zeros_like(self.robot_local_grasp_rot)

    def compute_reward(self, actions):
        # The rewards are reset buffers are set once the RL environment computes it.
        # The buffers are looked up by the .base.vec_task.py to do the PPO

        self.rew_buf[:], self.reset_buf[:], is_real_succ, is_test_succ = compute_robot_reward(
            self.reset_buf, self.progress_buf, self.actions, self.robot_grasp_pos, self.dist_reward_scale,
            self.action_penalty_scale, self.max_episode_length, self.voxel_pos, self.robot_net_cf,
            self.collision_reward_scale, self.num_envs, self.collision_penalty_type, self.voxel_size,
            self.test, self.real, self.pred_collision_prob, self.non_subst_collisions_yn, self.rupture_collisions_yn)

        # not really sure whh the > 10, to avoid cases where when no of steps = 0, reach is succ, guess from prev cycle.
        if self.test and self.test_steps_to_succ == self.max_episode_length and self.progress_buf[
            0].item() > 10 and is_test_succ.all():
            self.test_steps_to_succ = self.progress_buf[0].item()

        if is_real_succ.all():
            print("Real Target reached. Shutting down service .....")

            if self._arg_monitor_metric:
                print("REAL quantiles:")
                self.jm_monitor.quartiles()

            rki.shutdown_kinova_service()
            raise ValueError("Success: Real Target reached exiting ..... ")

    def check_for_nans(self, **obs_components):
        nan_row_env_ids = torch.Tensor([]).to(torch.int64).to(self.device)

        for c_name, c in obs_components.items():
            if torch.any(torch.isnan(c)):
                c_nan_indices = torch.where(torch.isnan(c).any(dim=1))[0]
                print(f"{c_name}: tensor contains NaN values at envs: {c_nan_indices}")
                nan_row_env_ids = torch.cat((nan_row_env_ids, c_nan_indices))

        nan_row_env_ids = nan_row_env_ids.unique()
        if nan_row_env_ids.numel() > 0:
            print(f"Combined unique row indices: {nan_row_env_ids}")
        return nan_row_env_ids

    def compute_tactile_obs(self):
        frame_no = self.gym.get_frame_count(self.sim)

        if self.stored_torque_set_count == 0:
            self.recent_reset_env_ids_classifier = torch.empty(0).to(self.device)

        if self.store_trg_torques and ((frame_no - 1) % store_frame_freq >= (store_frame_freq - store_frame_lag)):
            if self.stored_torque_set_count < self.max_trg_torques_to_store:

                robot_dof_torque_seq = self.robot_dof_torques.detach().clone().unsqueeze(0)
                robot_net_cf_seq = self.robot_net_cf.detach().clone().unsqueeze(0)
                # the list of env ids that were reset just before this frame.
                # in time series (i.e. if lag is used), the resets must be ignored, since time flow is broken by reset.
                robot_reset_envs = self.recent_reset_env_ids_classifier.detach().clone()

                print(f"===== Warning :frame_no: {frame_no} Storing Trg Torques: {self.stored_torque_set_count} ===== ")

                torch.save(robot_dof_torque_seq, torque_cf_seq_file("dof_torque", frame_no))
                torch.save(robot_net_cf_seq, torque_cf_seq_file("net_cf", frame_no))
                torch.save(robot_reset_envs, torque_cf_seq_file("reset_envs", frame_no))
                self.recent_reset_env_ids_classifier = torch.empty(0).to(self.device)

                self.stored_torque_set_count += 1

            else:
                self.store_trg_torques = False  # no more storing
                consts = {'store_frame_freq': store_frame_freq,
                          'store_frame_lag': store_frame_lag,
                          'num_envs': self.num_envs,
                          'brush_past_norm_cf': self.brush_past_norm_cf}

                with open(tactile_const_file, 'wb') as file:
                    pickle.dump(consts, file)

                raise ValueError("All Storing completed.")

        elif self.enable_joint_torque_obs:
            pass  # nothing to do at the moment.

    def apply_sim_classifier_proxy(self):
        """compute collision prob using a simulation proxy classifier. Computed directly from contact forces"""

        # magnitude of net contact force
        norm_coll_impact = torch.norm(self.robot_net_cf.view(self.num_envs, -1), p=2, dim=1)
        # A norm value of 20 = 3 newtons on all 16 links. Similarly 3N=> 20, 4N=>27, 5N=34
        self.non_subst_collisions_yn = norm_coll_impact < self.brush_past_norm_cf

        # if collision impact is too high, note those envs with a binary 1/0 to penalise later.
        self.rupture_collisions_yn = norm_coll_impact > 2.0 * self.brush_past_norm_cf

        # make a proxy real collision prob
        max_norm_impact_cf = 800  # This value should be around the 99th %ile; only for sim
        self.pred_collision_prob = norm_coll_impact / max_norm_impact_cf

    def apply_real_classifier(self):
        """compute collision prob using real classifier. """

        self.pred_collision_prob, _ = rki.real_collision_prob(loaded_model=self.loaded_classifier_model,
                                                              curr_dof_torque_mem=self.curr_dof_torque_mem,
                                                              classifier_type=rfc,
                                                              obs_vel_features=self.curr_obs_vel_recent_mem,
                                                              sliding_cmd_vel_features=self.curr_cmd_vel_recent_mem)

        self.non_subst_collisions_yn = self.pred_collision_prob < self.real_coll_prob_thresh

        # just place-holder no rupture computed for real.
        self.rupture_collisions_yn = torch.zeros_like(self.non_subst_collisions_yn).bool()
        self.curr_dof_torque_mem = self.curr_dof_torque_mem[-store_frame_lag:]

        # Monitoring
        curr_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{curr_timestamp_str}: Real collision prob: {self.pred_collision_prob}:{self.non_subst_collisions_yn},"
              f" threshold:{self.real_coll_prob_thresh}")

    def compute_real_observations(self):

        # equivalent to refresh_all_sim_tensors
        kinova_metrics = rki.fetch_all_real_tensors(device=self.device)

        hand_pos = kinova_metrics.hand_pos_t
        hand_rot = kinova_metrics.hand_rot_t

        # The robot_local_grasp_rot, robot_local_grasp_pos are fixed throughout, doesn't change during exec
        self.robot_grasp_rot[:], self.robot_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.robot_local_grasp_rot, self.robot_local_grasp_pos)

        self.robot_dof_pos = kinova_metrics.robot_dof_pos_t
        self.robot_dof_vel = kinova_metrics.robot_dof_vel_t
        self.robot_dof_torques = kinova_metrics.robot_dof_torque_t

        # use the current observation to decide the next vs the last desired state.
        # for real kinova this seems to be moe ideal.
        self.robot_dof_vel_targets[:, :self.num_robot_dofs] = self.robot_dof_vel.detach().clone()

        # create dof torque mem store for future
        dof_torque_t = self.robot_dof_torques.detach().clone()

        self.curr_dof_torque_mem = torch.cat((self.curr_dof_torque_mem, dof_torque_t), dim=0)
        self.curr_obs_vel_recent_mem = self.robot_dof_vel.detach().clone()  # for the classifier.

        # The observation space is composed of the robot joints’ normalized positions in the interval
        dof_pos_scaled = (2.0 * (self.robot_dof_pos - self.robot_dof_pos_lower_limits)
                          / (self.robot_dof_pos_upper_limits - self.robot_dof_pos_lower_limits) - 1.0)

        to_voxel_target = self.voxel_pos - self.robot_grasp_pos

        if self._arg_monitor_metric:
            self.jm_monitor.append(jp=dof_pos_scaled, jv=self.robot_dof_vel, jt=self.robot_dof_torques)

        if self.enable_joint_torque_obs:
            self.apply_real_classifier()  # classify collisions and get prob/yn values
            self.obs_buf = torch.cat((dof_pos_scaled,
                                      self.robot_dof_vel * self.dof_vel_scale,
                                      to_voxel_target,
                                      hand_rot,
                                      self.non_subst_collisions_yn.detach().clone().float().reshape(self.num_envs, -1)),
                                     dim=-1)

        else:
            raise ValueError("Nothing to do here")

        return self.obs_buf

    def compute_sim_observations(self):

        # Get current state after performing the simulation step (part of post_physics step)
        self.refresh_all_sim_tensors()

        # use the current observation to decide the next vs the last desired state.
        # for real kinova this seems to be moe ideal; i.e. switch from last desired state to current obs
        self.robot_dof_vel_targets[:, :self.num_robot_dofs] = self.robot_dof_vel.detach().clone()

        # This logic is to reset the environments which returns nan.
        nan_row_env_ids = self.check_for_nans(robot_dof_pos=self.robot_dof_pos,
                                              branch_poses=self.branch_poses,
                                              robot_dof_vel=self.robot_dof_vel)
        while nan_row_env_ids.numel() > 0:
            self.reset_idx(nan_row_env_ids)
            self.sim_instability_resets += nan_row_env_ids.numel()
            print("resetting envs:", nan_row_env_ids)

            for i in range(10):
                print(f"Skipping steps ....{i} : Total resets so far:{self.sim_instability_resets}..")
                self.render()
                self.gym.simulate(self.sim)
                self.refresh_all_sim_tensors()

                nan_row_env_ids = self.check_for_nans(robot_dof_pos=self.robot_dof_pos,
                                                      branch_poses=self.branch_poses,
                                                      robot_dof_vel=self.robot_dof_vel)

                if nan_row_env_ids.numel() == 0:
                    break

        if self.store_trg_torques or self.enable_joint_torque_obs:
            self.compute_tactile_obs()

        # get the current hand (panda link 7) pose.
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.voxel_pos = self.rigid_body_states[:, self.voxel_handle][:, 0:3]
        self.tree_root_pos = self.rigid_body_states[:, self.tree_root_handle][:, 0:3]

        # The robot_local_grasp_rot, robot_local_grasp_pos are fixed throughout, doesn't change during exec
        self.robot_grasp_rot[:], self.robot_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.robot_local_grasp_rot, self.robot_local_grasp_pos)

        self.robot_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.robot_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.robot_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.robot_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        # The observation space is composed of the robot joints’ normalized positions in the interval
        dof_pos_scaled = (2.0 * (self.robot_dof_pos - self.robot_dof_pos_lower_limits)
                          / (self.robot_dof_pos_upper_limits - self.robot_dof_pos_lower_limits) - 1.0)

        to_voxel_target = self.voxel_pos - self.robot_grasp_pos
        frame_no = self.gym.get_frame_count(self.sim)

        if self._arg_monitor_metric:
            self.jm_monitor.append(jp=dof_pos_scaled, jv=self.robot_dof_vel, jt=self.robot_dof_torques)

        if self.enable_joint_torque_obs:
            _dof_torque_tensors = self.robot_dof_torques.detach().clone()
            # self.curr_dof_torque_mem = torch.cat((self.curr_dof_torque_mem, _dof_torque_tensors), dim=0)
            # currently the max torques seems to be about 100
            _dof_torque_tensors_scaled = dof_torque_obs_scale * _dof_torque_tensors / 100

            self.apply_sim_classifier_proxy()
            # all input should go into obs_buf with size num_env x k
            # self.obs_buf = torch.cat((dof_pos_scaled,
            #                           self.robot_dof_vel * self.dof_vel_scale, to_voxel_target,
            #                           _dof_torque_tensors_scaled.reshape(self.num_envs, -1)),
            #                          dim=-1)

            if self.enable_symmetry_awareness:
                raise ValueError("Not tested yet")
            else:
                if self.collision_penalty_type.value >= CollisionPenalty.BINARY_PENALTY.value:
                    # =======
                    # PCAP
                    # =======
                    self.obs_buf = torch.cat((dof_pos_scaled,
                                              self.robot_dof_vel * self.dof_vel_scale,
                                              to_voxel_target,
                                              hand_rot,
                                              self.non_subst_collisions_yn.detach().clone().float().reshape(
                                                  self.num_envs, -1)),
                                             dim=-1)

                    # =======
                    # CPO
                    # =======
                    # self.obs_buf = torch.cat((dof_pos_scaled,
                    #                           self.robot_dof_vel * self.dof_vel_scale,
                    #                           to_voxel_target,
                    #                           hand_rot),
                    #                          dim=-1)
                    # =====================
                    # CPO + Joint Torques
                    # =====================
                    # self.obs_buf = torch.cat((dof_pos_scaled,
                    #                           self.robot_dof_vel * self.dof_vel_scale,
                    #                           to_voxel_target,
                    #                           hand_rot,
                    #                           _dof_torque_tensors_scaled.reshape(self.num_envs, -1)),
                    #                          dim=-1)
                else:
                    # Note: below is only for ablation only by removing the classifier.
                    # =====================
                    # Baseline PPO
                    # =====================
                    self.obs_buf = torch.cat((dof_pos_scaled,
                                              self.robot_dof_vel * self.dof_vel_scale,
                                              to_voxel_target,
                                              hand_rot),
                                             dim=-1)

                    # ========================
                    # Baseline PPO + Joint Torques
                    # =======================
                    # self.obs_buf = torch.cat((dof_pos_scaled,
                    #                           self.robot_dof_vel * self.dof_vel_scale,
                    #                           to_voxel_target,
                    #                           hand_rot,
                    #                           _dof_torque_tensors_scaled.reshape(self.num_envs, -1)),
                    #                          dim=-1)


        else:
            if frame_no % 300 == 0:
                print("Warning: Passing True points to state... ")
            # Note : The frame_no % 10 can be used if comparison is required. For regular RL just don't use it.
            if frame_no == init_steps_to_skip + 1:
                self.selected_branch_poses = self.branch_poses[:, self.random_branch_indices, :].detach().clone()
                self.selected_branch_poses_y_z = self.selected_branch_poses[:, :, 1:3]

            # only passing y-z pos of branches.
            self.obs_buf = torch.cat((dof_pos_scaled, self.robot_dof_vel * self.dof_vel_scale, to_voxel_target,
                                      self.selected_branch_poses_y_z.reshape(self.num_envs, -1)),
                                     dim=-1)

        if self.test:
            assert self.num_envs == 1, "For test have only 1 env else _net_cf_mag is invalid."
            _dof_torques_abs_sum = torch.abs(self.robot_dof_torques[:, 0:7]).sum().item()  # ignore gripper torques
            self.dof_torques_abs_sum_by_frame[-1].append(_dof_torques_abs_sum)

            _net_cf_mag = torch.norm(self.robot_net_cf.view(self.num_envs, -1), p=2, dim=1).item()  # verified
            self.net_cf_abs_sum_by_frame[-1].append(_net_cf_mag)

            # Check if any value is NaN
        if torch.any(torch.isnan(self.obs_buf)):
            print("===== Error: The obs_buf tensor contains NaN values. =====")
            print("robot_dof_pos >> ", self.robot_dof_pos, )
            self.check_for_nans(hand_pos=hand_pos, robot_dof_pos=self.robot_dof_pos, branch_poses=self.branch_poses,
                                robot_grasp_pos=self.robot_grasp_pos, to_voxel_target=to_voxel_target,
                                dof_pos_scaled=dof_pos_scaled, robot_dof_vel=self.robot_dof_vel)
            raise ValueError("The tensor contains NaN values.")

        return self.obs_buf

    def track_test_succ(self, env_ids_int32):

        selected_d2voxel = torch.norm(self.robot_grasp_pos - self.voxel_pos, p=2, dim=-1)[env_ids_int32]
        succ_mask = selected_d2voxel < self.voxel_size  # num of environments where gripper touches voxel.

        # Count the elements above the threshold using torch.sum
        succ_cnt = torch.sum(succ_mask).item()
        print("Success ?:", succ_cnt, "Total CF:", round(sum(self.net_cf_abs_sum_by_frame[-1]), 2),
              "Mean CF:", round(sum(self.net_cf_abs_sum_by_frame[-1]) / len(self.net_cf_abs_sum_by_frame[-1]), 2))
        print("No of steps taken to reach (succ/fail):", self.test_steps_to_succ)
        self.eval_succ_tracker.incr(succ_mask=[int(x) for x in succ_mask])
        self.eval_succ_tracker.add_steps_to_succ(steps_to_succ=[self.test_steps_to_succ])

    def reset_idx(self, env_ids):

        frame_no = self.gym.get_frame_count(self.sim)
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Store recent reset envs to skip these records for time series dof torque classifier in simulation
        self.recent_reset_env_ids_classifier = env_ids_int32
        # Store recent reset envs to reset robt_net_cf/previous forces to zero on reset.
        self.recent_reset_env_ids_frc = env_ids_int32.detach().clone()

        def _extract_lsystem_target_class(_rel_path):
            return os.path.splitext(os.path.basename(_rel_path))[0]

        def _finish_test():
            if len(self.eval_succ_tracker.succ_status) >= 50:
                # don't write out files if the number of tests are less than 50. So real executions will be skipped
                print("Saving succ stats .....")
                checkpoint_dir = rl_helper.extract_checkpoint_dir(cmd_args['checkpoint'])
                eval_metrics_args = (self.eval_dir, checkpoint_dir)

                if 'ablation' in self.lsystem_asset_root:
                    lsystem_target_class = _extract_lsystem_target_class(self.asset_rel_path)
                    eval_metrics_args = (self.eval_dir, checkpoint_dir, lsystem_target_class)

                with open(eval_dof_torques_abs_sum_fp(*eval_metrics_args), 'wb') as file:
                    pickle.dump(self.dof_torques_abs_sum_by_frame, file)

                with open(eval_net_cf_abs_sum_fp(*eval_metrics_args), 'wb') as file:
                    pickle.dump(self.net_cf_abs_sum_by_frame, file)

                with open(eval_succ_tracker_fp(*eval_metrics_args), 'wb') as file:
                    pickle.dump(self.eval_succ_tracker.get_counts(), file)

            print(self.eval_succ_tracker.get_counts())

            raise ValueError("all specified target voxel poses executed OR max test frames completed.")

        if self.real:
            # there is only one pose we execute at a time for real
            pose_idx, new_voxel_pose = next(self.eval_voxel_pos_gen, (1e3, None))
            if pose_idx > 1:
                print("Finished single real execution. Shutting down service .....")

                if self._arg_monitor_metric:
                    print("REAL quantile:")
                    self.jm_monitor.quartiles()

                rki.shutdown_kinova_service()
                raise ValueError("Timeout: Finished single real execution......")

            self.voxel_pos = torch.tensor(new_voxel_pose).to(self.device).unsqueeze(0)
            assert self.voxel_pos.shape == torch.Size([1, 3]), "voxel pos not read right"
            print(f"------------ \n{pose_idx}: {self.collision_penalty_type.name}: Setting voxel pose:", self.voxel_pos)
            self.curr_dof_torque_mem = torch.empty(0, self.num_dofs).to(self.device)

            # equivalent to refresh_all_sim_tensors
            kinova_metrics = rki.fetch_all_real_tensors(device=self.device)
            self.robot_dof_pos = kinova_metrics.robot_dof_pos_t
            self.robot_dof_vel = kinova_metrics.robot_dof_vel_t

            test_arm_pos = kinova_metrics.robot_dof_pos_t.detach().clone().squeeze(0)
            self.robot_dof_pos_targets[:, :self.num_robot_dofs] = test_arm_pos
            self.robot_dof_vel_targets[:, :self.num_robot_dofs] = 0
            # should be a 2D: 1x6 tensor([[4.6610, 3.4930, 1.6230, 4.2110, 0.9090, 1.5440)

        elif self.test:
            # in case of inference, reset voxel location as well for each iteration.
            assert self.root_state_tensor.shape == torch.Size([self.num_envs, 3, 13])

            if frame_no > init_steps_to_skip:
                self.track_test_succ(env_ids_int32)  # skip the first reset by using >

            for e in env_ids:
                if self.enable_eval_voxel_file:
                    pose_idx, new_voxel_pose = next(self.eval_voxel_pos_gen, (1e3, None))

                    if self._arg_monitor_metric and pose_idx > 1:
                        print("SIM Quartiles:")
                        self.jm_monitor.quartiles()
                        raise ValueError("stopping test: only one  test during quantile measurements.")

                    if new_voxel_pose is None:
                        _finish_test()

                    self.dof_torques_abs_sum_by_frame.append([])  # add to list of lists.
                    self.net_cf_abs_sum_by_frame.append([])
                    self.test_steps_to_succ = self.max_episode_length

                    self.root_state_tensor[e, 2, 0:3] = new_voxel_pose
                    print(f"------------ \n{pose_idx}: {self.collision_penalty_type.name}: Resetting voxel pose:",
                          self.root_state_tensor[env_ids, 2, 0:3])

                else:
                    if frame_no >= 2000:
                        _finish_test()
                    voxel_rand_pose = arm_random_loc_within_reach(arm_base_pose=self.robot_start_pose,
                                                                  arm_type=ArmType.kinova)
                    new_voxel_pose = to_torch([voxel_rand_pose.p.x, voxel_rand_pose.p.y, voxel_rand_pose.p.z])
                    self.root_state_tensor[e, 2, 0:3] = new_voxel_pose
                    print(f"------------ \nResetting voxel pose: {self.collision_penalty_type.name}:",
                          self.root_state_tensor[env_ids, 2, 0:3])

            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

        else:
            pass  # training

        # reset robot
        # Generate new random positions for the robot's degrees of freedom (DOFs).
        # These positions are sampled from a uniform distribution centered around the default DOF positions
        # The magnitude of the change is controlled by a factor of 0.25.
        #  Clamp the generated positions to ensure they are within the specified lower and upper dof limits.

        if not self.real:
            arm_pos = tensor_clamp(
                self.kinova_default_dof_pos.unsqueeze(0) + 0.25 * (
                        torch.rand((len(env_ids), self.num_robot_dofs), device=self.device) - 0.5),
                self.robot_dof_pos_lower_limits, self.robot_dof_pos_upper_limits)

            test_arm_pos = self.kinova_default_dof_pos.detach().clone()

            #  Update the state of Robot's DOF positions, velocities, and DOF targets for the specified env IDs
            #  with the newly generated positions.

            self.robot_dof_pos[env_ids, :] = test_arm_pos if self.test else arm_pos
            self.robot_dof_vel[env_ids, :] = torch.zeros_like(self.robot_dof_vel[env_ids])

            self.robot_dof_pos_targets[env_ids, :self.num_robot_dofs] = test_arm_pos if self.test else arm_pos
            self.robot_dof_vel_targets[env_ids, :self.num_robot_dofs] = 0

            # reset tree
            self.tree_dof_state[env_ids, :] = self.tree_default_dof_state[env_ids, :].detach().clone()
            # self.tree_dof_state[env_ids, :] = torch.zeros_like(self.tree_dof_state[env_ids])

            # I think the 2 is because of both tree and robot
            multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()

            # The dof_state gets updated when we update robot_dof_pos, tree_dof_state , e.t.c
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                  gymtorch.unwrap_tensor(self.dof_state),
                                                  gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

            self.progress_buf[env_ids] = 0
            self.prev0_robot_net_cf[env_ids, :, :] = 0
            self.prev1_robot_net_cf[env_ids, :, :] = 0
            self.prev2_robot_net_cf[env_ids, :, :] = 0
            self.robot_net_cf[env_ids, :, :] = 0

        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        # TODO: move this to conf and the validation outside instead of doing each time.
        real_write_freq = 60.  # frequency to write, in kinova this is close to 100Hz
        real_write_delay = 1.0 / real_write_freq
        # ideal value is 3,4,5. With stagger 5 and freq 60, the read freq is about 12Hz
        real_stagger_factor = 6

        # if you write too slow, the read/fetch will skip readings.
        # if you write too fast, the read/fetch have duplicate readings.
        assert 30 > real_write_freq / real_stagger_factor >= 10, "Kinova read freq is 10Hz"

        # No of steps to split the actions to
        # hopefully the higher the factor, the slower the arm gets.
        # 60 and 3-5 works; but combined should not go below 10HZ, because the read is in 10Hz

        if self.action_attempts == 0:
            self.action_start_time = time.time()

        # pre_physics_step:Actions.shape torch.Size([num_envs, 9])
        # 9 dof for franka, 6 for kinova
        self.actions = actions.clone().to(self.device)

        if self.transferable_to_real:
            if self.real:
                self.apply_real_staggered_vel_actions(real_write_delay, real_stagger_factor)
            else:  # train & test with lower read-write frequency
                self.apply_sim_staggered_vel_actions(real_write_delay, real_stagger_factor)
        else:
            self.apply_sim_vel_actions()  # follow sim freq without staggering.

    def apply_real_staggered_vel_actions(self, real_write_delay, real_stagger_factor):
        """For sim 2 real for velocity divide the REAL targets into smaller chunks
           so that you can meet the frequency requirement of Kinova"""

        # Note: for dof pos, reduce action_scale, you get shorter strides & less jerky motion
        delta_vel_target = self.robot_dof_speed_scales * self.dt * self.actions * self.action_scale * (
                1 / real_stagger_factor)

        for _chunk_idx in range(real_stagger_factor):
            vel_targets = self.robot_dof_vel_targets[:, :self.num_robot_dofs] + delta_vel_target

            self.robot_dof_vel_targets[:, :self.num_robot_dofs] = tensor_clamp(
                vel_targets, -1 * self.robot_dof_vel_max_limits, self.robot_dof_vel_max_limits)

            target_dof_vel = self.robot_dof_vel_targets.detach().clone().squeeze(0)
            robot_resp = rki.set_real_robot_dof_vel(target_dof_vel)

            assert robot_resp == robot_domain.CommStatus.SUCCESS

            # to match the targeted freq for kinova.
            self.action_attempts += 1
            self.action_end_time = time.time()
            elapsed_time = self.action_end_time - self.action_start_time
            remaining_delay = real_write_delay - elapsed_time

            # If there's remaining time, delay the loop
            if remaining_delay > 0:
                time.sleep(remaining_delay)

            if self.action_attempts % 200 in list(range(1, real_stagger_factor + 1)):
                # This block is just to monitor the frequency of operation every n actions.
                if remaining_delay < 0:
                    print(f"Warning: {self.action_attempts} Iteration took longer than expected time")

                re_elapsed_time = round(time.time() - self.action_start_time, 3)
                freq_actions = round(1 / re_elapsed_time, 3)

                freq_msg = f"Freq (all should be ~60:): {freq_actions} Hz"
                print(f"Real: Idx:{_chunk_idx}: Prog:{self.progress_buf}: Elapsed:{re_elapsed_time} sec, {freq_msg}")

            self.action_start_time = time.time()  # re-initialise start time after each time step.

        # the commanded velocity for classifier; computed as  sum of targets if all chunks were applied together,
        self.curr_cmd_vel_recent_mem = self.robot_dof_vel_targets[:, :self.num_robot_dofs].detach().clone()

    def apply_sim_staggered_vel_actions(self, real_write_delay, real_stagger_factor):
        """For sim 2 real for velocity we divide the targets into smaller chunks; therefore stagger sims to ensure that
        the training sim_action block and the real_action block are approximately executed in the same time."""

        sim_write_delay = real_write_delay * real_stagger_factor

        delta_vel_target = self.robot_dof_speed_scales * self.dt * self.actions * self.action_scale

        vel_targets = self.robot_dof_vel_targets[:, :self.num_robot_dofs] + delta_vel_target

        # ensure that the robot's DOFs do not move beyond safe or valid ranges
        self.robot_dof_vel_targets[:, :self.num_robot_dofs] = tensor_clamp(
            vel_targets, -1 * self.robot_dof_vel_max_limits, self.robot_dof_vel_max_limits)

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_vel_targets))

        # to match the targeted freq for kinova.
        self.action_attempts += 1
        self.action_end_time = time.time()
        elapsed_time = self.action_end_time - self.action_start_time
        remaining_delay = sim_write_delay - elapsed_time

        # If there's remaining time, delay the loop
        if remaining_delay > 0:
            time.sleep(remaining_delay)

        if self.action_attempts % 200 == 0:
            # This block is just to monitor the frequency of operation every n actions.
            if remaining_delay < 0:
                print(f"Warning: {self.action_attempts} Iteration took longer than expected time")

            re_elapsed_time = round(time.time() - self.action_start_time, 3)
            freq_actions = round(1 / re_elapsed_time, 3)
            freq_msg = f"Freq (should be ~10Hz:): {freq_actions} Hz"

            print(f"Sim: Acts: {self.action_attempts}: Elapsed:{re_elapsed_time} sec, {freq_msg}")
            assert freq_actions > 7., "Too low frequency in sim"

        self.action_start_time = time.time()  # re-initialise start time after each time step.

    def apply_sim_vel_actions(self):
        """ Legacy function to be used in non-sim2real cases"""

        # robot_dof_targets is initially set to 0 with shape (self.num_envs, self.num_dofs)
        vel_targets = self.robot_dof_vel_targets[:,
                      :self.num_robot_dofs] + self.robot_dof_speed_scales * self.dt * self.actions * self.action_scale

        # ensure that the robot's DOFs do not move beyond safe or valid ranges
        self.robot_dof_vel_targets[:, :self.num_robot_dofs] = tensor_clamp(
            vel_targets, -1 * self.robot_dof_vel_max_limits, self.robot_dof_vel_max_limits)

        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.robot_dof_vel_targets))

        self.action_attempts += 1

        assert self.transferable_to_real is False, "Inappropriate for sim2real"

    # implement computations performed after stepping the physics simulation,
    # for e.g. computing rewards and observations.
    def post_physics_step(self):

        # This is a tensor of size [num_envs]
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        if self.real:
            # invoke in case of test with real arm
            self.compute_real_observations()
        else:
            # invoke in case of train & test in sim
            self.compute_sim_observations()

        self.compute_reward(self.actions)


#####################################################################
###=========================jit functions=========================###
#####################################################################


# noinspection PyTypeChecker
@torch.jit.script
def compute_robot_reward(
        reset_buf, progress_buf, actions, robot_grasp_pos, dist_reward_scale, action_penalty_scale,
        max_episode_length, voxel_pos, robot_net_cf, collision_reward_scale, num_envs, collision_penalty_type,
        voxel_size, is_test, is_real, pred_collision_prob, non_subst_collisions_yn, rupture_collisions_yn
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, Tensor, Tensor, float, int, CollisionPenalty, float, bool, bool, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    d2voxel = torch.norm(robot_grasp_pos - voxel_pos, p=2, dim=-1)

    # boolean indicating if target reached in case of real executing.
    is_real_succ = (d2voxel < voxel_size).all() if is_real else torch.tensor([False])
    is_test_succ = (d2voxel < voxel_size).all() if is_test else torch.tensor([False])

    # compute voxel distance rewards.
    voxel_dist_reward = 1.0 / (1.0 + d2voxel ** 2)
    voxel_dist_reward *= voxel_dist_reward

    # 2 rounds of bonus
    voxel_dist_reward = torch.where(d2voxel <= voxel_size, voxel_dist_reward * 2, voxel_dist_reward)
    voxel_dist_reward = torch.where(d2voxel <= (voxel_size / 2), voxel_dist_reward * 2, voxel_dist_reward)
    voxel_dist_reward = torch.where(d2voxel <= (voxel_size / 4), voxel_dist_reward * 2, voxel_dist_reward)

    # regularization on the actions (summed for each environment). This is tho enable smooth movement.
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # sum rewards but scale them first.
    voxel_rewards = dist_reward_scale * voxel_dist_reward - action_penalty_scale * action_penalty

    if collision_penalty_type == CollisionPenalty.NO_PENALTY:
        pass
    elif collision_penalty_type == CollisionPenalty.BINARY_PENALTY:

        # Product of rewards: For product, the coll reward must be > 1 (i.e > 100 when collision_reward_scale=0.01)
        collision_reward = torch.where(non_subst_collisions_yn, 120, 0)
        voxel_rewards = voxel_rewards * collision_reward_scale * collision_reward

        # negative reward for too high rupture causing collisions.
        rupture_neg_reward = torch.where(rupture_collisions_yn, -120, 0)
        voxel_rewards = voxel_rewards + collision_reward_scale * rupture_neg_reward

    elif collision_penalty_type == CollisionPenalty.DECAY_PENALTY:

        coll_impact_reward = 0.18 / (0.35 + pred_collision_prob ** 2)
        coll_impact_reward *= coll_impact_reward

        # rounds of bonus
        coll_impact_reward = torch.where(pred_collision_prob <= 1 / 8, coll_impact_reward * 1.8, coll_impact_reward)
        coll_impact_reward = torch.where(pred_collision_prob <= 1 / 16, coll_impact_reward * 1.8, coll_impact_reward)
        coll_impact_reward = torch.where(pred_collision_prob <= 1 / 64, coll_impact_reward * 1.8, coll_impact_reward)

        voxel_rewards = voxel_rewards + coll_impact_reward

    else:
        raise ValueError("Not implemented.")

    if not (is_test or is_real):  # during test use only max episode length for reset
        reset_buf = torch.where(d2voxel < (voxel_size / 2), torch.ones_like(reset_buf), reset_buf)

    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    if not (is_test or is_real):
        resettable_norm_impact_cf = 10000
        resettable_coll_impact = torch.norm(robot_net_cf.view(num_envs, -1), p=2, dim=1)
        # reset environment if net cf is more than 20000
        reset_buf = torch.where(resettable_coll_impact > resettable_norm_impact_cf,
                                torch.ones_like(reset_buf), reset_buf)

    return voxel_rewards, reset_buf, is_real_succ, is_test_succ


# PyTorch's Just-In-Time (JIT) compilation for improved performance.
@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos):
    # type: (Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]

    # takes the variable current position of the hand (origin of panda_link7), combines with the
    # fixed offset to the  grasp point (mid of finger) to form the current position of the grasp point (mid of fingers)
    global_robot_rot, global_robot_pos = tf_combine(
        hand_rot, hand_pos, robot_local_grasp_rot, robot_local_grasp_pos)

    return global_robot_rot, global_robot_pos
