"""
1. Initialise Isaac Gym
2. Load domain randomised lsystem trees.
3. For each tree find out the degree of rotation so that the flexible dof-links and the robot is the closest.
4. Also weed out the trees whose average flexible dof-link pose is not reachable for the end-effector

Invocation

python source/reinforcement/tactile/tree_base_rotation_calculator.py
"""

import sys
import os
import math
import numpy as np
import pickle
import copy

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import torch

spacing = 2.0
use_gpu_pipeline = True
enable_viewer = False
dynamics_by_beam_deflection = False  # currently beam deflection is not supported here

# repeat the tree in parallel environments to check the best rotation where the dofs are closest to Franka
tree_repeats = 10

spot_path = os.getenv('spot_path')
source_dir = f"{spot_path}/source/"

if source_dir not in sys.path:
    sys.path.append(source_dir)

from common.helpers import futils
from reinforcement.ige.helpers import pos_locator
from reinforcement.ige.helpers import physics_helper, rl_helper

device = torch.device('cuda' if torch.cuda.is_available() and use_gpu_pipeline else 'cpu')


def set_up_simulation():
    # Initialize gym
    gym = gymapi.acquire_gym()

    # Parse arguments
    args = gymutil.parse_arguments(description="Tree Loader")

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 15
        sim_params.flex.relaxation = 0.75
        sim_params.flex.warm_start = 0.8
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = use_gpu_pipeline

    sim = gym.create_sim(args.compute_device_id,
                         args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        quit()

    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties()) if enable_viewer else None
    if enable_viewer and viewer is None:
        quit()

    # Add ground plane
    plane_params = gymapi.PlaneParams()

    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up!
    plane_params.distance = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)

    return gym, sim, viewer


def create_envs(num_envs, num_tree_types, asset_root, asset_rel_path, gym, sim, viewer):
    repeated_file_indices = np.repeat(np.arange(num_tree_types), tree_repeats)

    # gt_tree_asset_files count is equal to num_envs
    tree_asset_files = [asset_rel_path.format(r_idx=r_idx) for r_idx in repeated_file_indices]

    tree_asset_options = gymapi.AssetOptions()
    tree_asset_options.fix_base_link = True
    tree_asset_options.armature = 0.01
    tree_asset_options.collapse_fixed_joints = True

    tree_asset_options.override_com = True
    tree_asset_options.override_inertia = True

    tree_assets = []
    for i in range(num_envs):
        if i % 100 == 0:
            print(f"Processing env => {i} of {num_envs}")

        # Note this is where the segmentation fault happens for large/too many files.
        _tree_asset = gym.load_asset(sim, asset_root, tree_asset_files[i], tree_asset_options)
        tree_assets.append(_tree_asset)

    rot_quaternions = []  # Rotation Array
    for i in range(0, tree_repeats):
        turn = i * (2 / tree_repeats)  # rotate by pi/2
        rot_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), turn * math.pi)
        rot_euler = rot_quat.to_euler_zyx()
        print(f"Rotating options {i}:", [math.degrees(r) for r in rot_euler])
        rot_quaternions.append(rot_quat)

    all_rot_quaternions = [copy.deepcopy(rot_quaternions) for _ in range(num_tree_types)]

    # set up the env grid
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)
    num_per_row = int(math.sqrt(num_envs))

    # cache some common handles for later use
    envs = []
    actor_handles = []

    # create and populate the environments
    for env_i in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, num_per_row)
        envs.append(env)

        gt_tree_pose = gymapi.Transform()
        gt_tree_pose.p = gymapi.Vec3(0, 0, 0.0)

        rot_row, rot_col = env_i // tree_repeats, env_i % tree_repeats
        gt_tree_pose.r = copy.deepcopy(all_rot_quaternions[rot_row][rot_col])

        actor_handle = gym.create_actor(
            env, tree_assets[env_i], gt_tree_pose, f"GroundTruthTree_{env_i}", env_i, 1)

        actor_handles.append(actor_handle)

    assert len({*actor_handles}) == 1, "Invalid Actor Handles Created"

    for env_i in range(num_envs):
        link_dict = gym.get_asset_rigid_body_dict(tree_assets[env_i])
        assert link_dict['world'] == 0, 'World and the collapsed joints must be 0 indexed'

    if enable_viewer:
        cam_pos = gymapi.Vec3(4, 3, 2)
        cam_target = gymapi.Vec3(-4, -3, 0)
        num_per_row = int(math.sqrt(num_envs))
        middle_env = envs[num_envs // 2 + num_per_row // 2]
        gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

    return all_rot_quaternions, tree_assets, actor_handles, envs


def simulate_rotation(num_envs, num_tree_types, asset_root, all_rot_quaternions,
                      tree_assets, actor_handles, envs, gym, sim, viewer, franka_knee_position, franka_reach):
    gym.prepare_sim(sim)
    _rb_states = gym.acquire_rigid_body_state_tensor(sim)
    rb_states = gymtorch.wrap_tensor(_rb_states)
    rb_states = rb_states.view(num_envs, -1, 13)

    # link 1 is the dof_root as the other bodies have collapsed.
    dof_link_positions = rb_states[:, 1:, 0:3]

    for env_i in range(num_envs):
        tree_dof_props = gym.get_asset_dof_properties(tree_assets[env_i])
        num_tree_dofs = gym.get_asset_dof_count(tree_assets[env_i])

        for dof_i in range(num_tree_dofs):
            noise_std = 1.
            if dynamics_by_beam_deflection:
                raise ValueError("not implemented yet.")
            else:
                # use rudimentary policy for test and train
                branch_level = rl_helper.branch_level(gym.get_asset_dof_name(tree_assets[env_i], dof_i))
                kp, kd = physics_helper.rud_deflection_param(branch_level=branch_level, base_kp=400,
                                                             noise_std=noise_std)
                tree_dof_props['stiffness'][dof_i] = kp
                tree_dof_props['damping'][dof_i] = kd
                tree_dof_props['friction'][dof_i] = 0.01  # some low value

        gym.set_actor_dof_properties(envs[env_i], actor_handles[env_i], tree_dof_props)

    for step_idx in range(10):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        dof_link_mean_pos = torch.mean(dof_link_positions, dim=1)  # mean of body pos for all bodies except world.

        if viewer:

            gym.clear_lines(viewer)
            for j in range(num_envs):
                cube_lines = pos_locator.draw_cube(center=dof_link_mean_pos[j].cpu().numpy(),
                                                   side_length=0.05)

                for i in range(len(cube_lines)):
                    gym.add_lines(viewer, envs[j], 1,
                                  [cube_lines[i][0][0], cube_lines[i][0][1], cube_lines[i][0][2],
                                   cube_lines[i][1][0], cube_lines[i][1][1], cube_lines[i][1][2]],
                                  [0.0, 0.0, 1.0])

        # update viewer
        if enable_viewer:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, False)
            gym.sync_frame_time(sim)

    print("Removing invalid trees....")
    # Calculate Euclidean distance b/w expected franka position and the pliable dof root
    dof_link_mean_pos = torch.mean(dof_link_positions, dim=1)  # mean of body pos for all bodies except world.

    dof_link_mean_d2_franka = torch.sqrt(torch.sum((dof_link_mean_pos - franka_knee_position) ** 2, dim=1))

    assert dof_link_mean_d2_franka.shape == torch.Size(
        [num_tree_types * tree_repeats]), f"Bad dim:{dof_link_mean_d2_franka.shape}"

    best_rot_indices = torch.argmin(dof_link_mean_d2_franka.view(num_tree_types, -1), dim=1)
    best_dof_link_mean_d2_franka, _ = torch.min(dof_link_mean_d2_franka.view(num_tree_types, -1), dim=1)
    # consider unreachable if the mean dof link is 1.3 times beyond the franka reach
    dof_link_mean_reachable = best_dof_link_mean_d2_franka < franka_reach * 1.2

    print(
        f"Valid configs:{dof_link_mean_reachable.sum()}, invalid configs: {num_tree_types - dof_link_mean_reachable.sum()}")
    # assert num_tree_types - dof_link_mean_reachable.sum() < 0.25 * num_tree_types, "Too many invalid trees."

    tree_alignment_quats = [all_rot_quaternions[rot_row][rot_col] for rot_row, rot_col in
                            enumerate(best_rot_indices.tolist())]

    return dof_link_mean_reachable, tree_alignment_quats


def shutdown(gym, sim, viewer):
    if viewer:
        gym.destroy_viewer(viewer)

    gym.destroy_sim(sim)


def save_valid_trees(asset_root_raw, asset_rel_path, unreachable_tree_indices, num_tree_types,
                     target_clean_path_suffix, ternary_class):
    assert asset_root_raw[-1] != '/'
    asset_root_parent = os.path.dirname(asset_root_raw)  # the randomised folder
    asset_root_clean = os.path.join(asset_root_parent, target_clean_path_suffix)

    # cleanup target directory
    futils.cleanup_sub_folders_n_files(_dir=asset_root_clean, folder_prefix=ternary_class,
                                       file_types=(".png", ".jpg", ".npy"))

    valid_tree_asset_files = [asset_rel_path.format(r_idx=r_idx)
                              for r_idx in range(num_tree_types) if r_idx not in unreachable_tree_indices]

    for nr_idx, _rel_path in enumerate(valid_tree_asset_files):
        target_file_name = asset_rel_path.format(r_idx=nr_idx)

        futils.copy_file(source_file=os.path.join(asset_root_raw, _rel_path),
                         target_folder=os.path.join(asset_root_clean,
                                                    os.path.dirname(target_file_name)))
        futils.copy_file(source_file=os.path.join(asset_root_raw, _rel_path.replace('.urdf', '_meta.pkl')),
                         target_folder=os.path.join(asset_root_clean,
                                                    os.path.dirname(target_file_name)))

    return asset_root_clean


def alignment_calculator(asset_root_raw, asset_rel_path, target_clean_path_suffix,
                         franka_sp_tup, franka_knee_ht, franka_reach, ternary_class):
    """
        franka_sp_tup: Starting position tuple
        franka_knee_ht = height from the floor to the knee. The reachability starts from the knee
        franka_reach = Max reach from the knee to weed out invalid trees.
        Refer: https://www.generationrobots.com/media/panda-franka-emika-datasheet.pdf

    """
    num_tree_types = futils.count_sub_folders(asset_root_raw)  # types of trees available
    num_envs = num_tree_types * tree_repeats

    franka_kp_tup = franka_sp_tup[:-1] + (franka_sp_tup[-1] + franka_knee_ht,)
    franka_knee_position = torch.tensor(franka_kp_tup, dtype=torch.float32, device=device).unsqueeze(0)

    gym, sim, viewer = set_up_simulation()
    all_rot_quaternions, tree_assets, actor_handles, envs = create_envs(num_envs, num_tree_types, asset_root_raw,
                                                                        asset_rel_path, gym, sim, viewer)

    dof_link_mean_reachable, tree_alignment_quats = simulate_rotation(num_envs, num_tree_types, asset_root_raw,
                                                                      all_rot_quaternions, tree_assets, actor_handles,
                                                                      envs, gym, sim, viewer,
                                                                      franka_knee_position, franka_reach)

    reachable_tree_indices = torch.nonzero(dof_link_mean_reachable).squeeze().tolist()
    unreachable_tree_indices = torch.nonzero(~dof_link_mean_reachable).squeeze().tolist()

    reachable_tree_indices = [reachable_tree_indices] if not isinstance(
        reachable_tree_indices, list) else reachable_tree_indices
    unreachable_tree_indices = unreachable_tree_indices if isinstance(
        unreachable_tree_indices, list) else [unreachable_tree_indices]

    print("reachable_tree_indices:", reachable_tree_indices)
    print("raw/raw_test => train/test map:")
    for __idx, _rch_idx in enumerate(reachable_tree_indices):
        print(_rch_idx, " => ", __idx, end=', ')

    asset_root_clean = save_valid_trees(asset_root_raw, asset_rel_path, unreachable_tree_indices,
                                        num_tree_types, target_clean_path_suffix, ternary_class)

    valid_tree_alignment_quats = [v for _i, v in enumerate(tree_alignment_quats) if _i not in unreachable_tree_indices]
    assert len(valid_tree_alignment_quats) == num_tree_types - len(unreachable_tree_indices)

    tree_alignment_quats_file = os.path.join(asset_root_clean, 'tree_alignment_quats.pkl')
    with open(tree_alignment_quats_file, 'wb') as file:
        pickle.dump(valid_tree_alignment_quats, file)

    shutdown(gym, sim, viewer)
    print("Completed Storing best rotation alignment for the assets.")
    return tree_alignment_quats


if __name__ == "__main__":
    _ternary_class = 'ta'  # ta, tb, tc, td only . ts represents non l-system class (simple tree)
    assert _ternary_class in ['ta', 'tb', 'tc', 'td', 'ts']

    # _asset_root_raw_rel = f"simulation/lsystems/three/urdf/tree/gen/ablation/std_0.1/{_ternary_class}/raw"
    # _asset_root_raw_rel = f"simulation/basic/urdf/tree/gen/randomised/{_ternary_class}/raw"
    _asset_root_raw_rel = f"source/simulation/deformable/three/urdf/tree/gen/pliable00/{_ternary_class}/raw"

    _asset_root_raw = os.path.join(source_dir, _asset_root_raw_rel)
    assert _asset_root_raw.endswith("/raw") or _asset_root_raw.endswith("/raw_test"), "Path does not end with '/raw'"
    _target_clean_path_suffix = 'train' if _asset_root_raw.endswith("/raw") else 'test'

    user_input = input(f" This is the old version to be used with PCAP only. "
                       f" Continue? (yes/no): ").strip().lower()
    if user_input not in ('yes', 'no'):
        print("Please enter 'yes' or 'no'.")

    if user_input != 'yes':
        quit()

    user_input = input(f" {_target_clean_path_suffix} files will be re-written for class {_ternary_class}."
                       f" Continue? (yes/no): ").strip().lower()
    if user_input not in ('yes', 'no'):
        print("Please enter 'yes' or 'no'.")

    if user_input != 'yes':
        quit()

    # don't use / at the end of raw path.

    _asset_rel_path = f"{_ternary_class}_" + "{r_idx}/" + f"ternary_{_ternary_class[-1]}.urdf"
    _franka_sp_tup = (1.0, 0.0, 0.0)
    # height from the floor to the knee. https://frankaemika.github.io/docs/_images/dh-diagram.png
    _franka_knee_ht = 0.3330
    _franka_reach = 0.855

    alignment_calculator(_asset_root_raw, _asset_rel_path, _target_clean_path_suffix,
                         _franka_sp_tup, _franka_knee_ht, _franka_reach, _ternary_class)
