"""
This file is a modified version of <IsaacGym>/python/examples/franka_cube_ik_osc.py

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf

Invocation

python source/reinforcement/tactile/franka_dof_torque_examples.py --num_envs 1
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import sys
import os
import math
import numpy as np
import torch
import pickle

spot_path = os.getenv('spot_path')
source_dir = f"{spot_path}/source/"

if source_dir not in sys.path:
    sys.path.append(source_dir)

_set = 'set_p1'

from common.helpers import futils

# The directory to store forces, which can be inspected through a jupyter notebook.
_force_store_dir = f"{spot_path}/notebooks/work2/data/tactile/raw/{_set}/out"
_dof_seq_file = f"{_force_store_dir}/dof_torque_seq.pt"
_net_cf_file = f"{_force_store_dir}/franka_net_seq.pt"
_tactile_const_file = f"{_force_store_dir}/tactile_constants.pkl"


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
            kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
            (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--num_envs", "type": int, "default": 2, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="Franka Jacobian Inverse Kinematics (IK) + Operational Space Control (OSC) Example",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.05

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = "source/common/assets"

# create table asset
table_dims = gymapi.Vec3(0.6, 1.0, 0.1)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
box_color = gymapi.Vec3(0.0, 0.0, 1.0)  # Red color

# create box asset
box_size = 0.045
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load franka asset
franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
# Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system
asset_options.flip_visual_attachments = True
franka_asset = gym.load_asset(sim, asset_root, franka_asset_file, asset_options)

# configure franka dofs
franka_dof_props = gym.get_asset_dof_properties(franka_asset)

# lower and upper joint limit in radians.
franka_lower_limits = franka_dof_props["lower"]
franka_upper_limits = franka_dof_props["upper"]
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.3 * (franka_upper_limits + franka_lower_limits)

print("franka_lower_limits >>", franka_lower_limits)
print("franka_upper_limits >>", franka_upper_limits)
print("franka_mids >>", franka_mids)

# from https://github.com/NVlabs/oscar/blob/main/oscar/cfg/train/agent/franka.yaml
conf_dof_stiffness = 400.
conf_dof_damping = 40.
conf_dof_friction = 0.01

if controller == "ik":
    franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:7].fill(conf_dof_stiffness)
    franka_dof_props["damping"][:7].fill(conf_dof_damping)
    franka_dof_props["friction"][:7].fill(conf_dof_friction)
else:  # osc
    raise ValueError("Not Implemented")
# grippers
franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
# grippers fixed config
franka_dof_props["stiffness"][7:].fill(800.0)
franka_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
franka_num_dofs = gym.get_asset_dof_count(franka_asset)

print("franka_num_dofs", franka_num_dofs)
default_dof_pos = np.zeros(franka_num_dofs, dtype=np.float32)

# The position of each dof defined in the urdf is taken to be 0.
# so you can configure any other position to be the starting pos (called default in this example.)
default_dof_pos[:7] = franka_mids[:7]
# keep grippers open
default_dof_pos[7:] = franka_upper_limits[7:]

# state = (position, velocity)
default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
print("default_dof_state before ", default_dof_state)
default_dof_state["pos"] = default_dof_pos

print("default_dof_pos", default_dof_pos)
print("default_dof_state after", default_dof_state)

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
franka_link_dict = gym.get_asset_rigid_body_dict(franka_asset)

# panda has 11 links and 9 joints.
print("franka_link_dict, ", franka_link_dict)  # hand is just below the fingers
franka_hand_index = franka_link_dict["panda_hand"]

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

franka_pose = gymapi.Transform()
franka_pose.p = gymapi.Vec3(0, 0, 0)

table_pose = gymapi.Transform()
# table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)
table_pose.p = gymapi.Vec3(0.6, 0.0, 0.5 * table_dims.z)

# Create different types of obstacles & obstacle poses
# obstacle_dims = gymapi.Vec3(0.2, 0.5, 0.3) # obstacle 1
# obstacle_dims = gymapi.Vec3(0.05, 0.5, 0.5) # # obstacle 2
# obstacle_dims = gymapi.Vec3(0.5, 0.05, 0.5) # # obstacle 3
obstacle_dims = gymapi.Vec3(0.5, 0.05, 0.9)  # # obstacle 4

obstacle_asset_options = gymapi.AssetOptions()
obstacle_asset_options.fix_base_link = True
obstacle_asset = gym.create_box(sim, obstacle_dims.x, obstacle_dims.y, obstacle_dims.z, obstacle_asset_options)
obstacle_color = gymapi.Vec3(1.0, 0.8, 0)  # Red color

obstacle_pose = gymapi.Transform()
# obstacle_pose.p = gymapi.Vec3(0.3, 0.0, (0.5 * table_dims.z + 0.5 * obstacle_dims.z)) # obstacle 1
# obstacle_pose.p = gymapi.Vec3(0.15, 0.0, (0.5 * table_dims.z + 0.5 * obstacle_dims.z)) # obstacle 2
# obstacle_pose.p = gymapi.Vec3(0.3, -0.25, (0.5 * table_dims.z + 0.5 * obstacle_dims.z)) # obstacle 3
obstacle_pose.p = gymapi.Vec3(0.3, -0.28, (0.5 * table_dims.z + 0.5 * obstacle_dims.z))  # obstacle 4

box_pose = gymapi.Transform()

envs = []
box_idxs = []
hand_idxs = []
init_pos_list = []
init_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 1)

    # add obstacle
    obstacle_handle = gym.create_actor(env, obstacle_asset, obstacle_pose, "obstacle", i, 2)
    gym.set_rigid_body_color(env, obstacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, obstacle_color)

    # add box at a random x,y position , rotation & color.
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    # box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3) # obstacle 1 & 2
    box_pose.p.y = table_pose.p.y - 0.4  # obstacle 3
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 3)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add franka
    franka_handle = gym.create_actor(env, franka_asset, franka_pose, "franka", i, 4)

    # enable dof force sensors explicitly.
    gym.enable_actor_dof_force_sensors(env, franka_handle)

    # set dof properties, stiffness, damping e.t.c
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # set initial dof states.
    # i.e. where you want each joint to be
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    # same pos as targets too.. so that it will come back to that position in case of an error.
    # if we don't set this, default targets will be 0, i.e the structure defined by the urdf
    gym.set_actor_dof_position_targets(env, franka_handle, default_dof_pos)

    # get initial hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

num_franka_bodies = gym.get_asset_rigid_body_count(franka_asset)
num_bodies = 1 + 1 + 1 + num_franka_bodies  # table/obstacle/box/franka

# point camera at middle env, this is over-ridden later.
cam_pos = gymapi.Vec3(4, 3, 2)
cam_target = gymapi.Vec3(-4, -3, 0)

middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
# print("corner_coord", corner_coord)
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# Jacobian gives the small change in end effector position, when each of the joint angles changes by a small amount
# Essential Jacobian is the differential relation b/w end effector position and joint displacement.
# get jacobian tensor
# for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

#  the shape of the Jacobian tensor will be (num_envs, num_links, 6 (linear/angular along xyz), num_dofs)
# jacobian entries corresponding to franka hand
j_eef = jacobian[:, franka_hand_index - 1, :, :7]

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]  # only need elements corresponding to the franka arm

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)

_dof_forces = gym.acquire_dof_force_tensor(sim)
dof_forces = gymtorch.wrap_tensor(_dof_forces)
dof_forces_seq = torch.empty(0, 9).to(device)
franka_net_cf_seq = torch.empty(0, num_franka_bodies, 3).to(device)

_net_cf = gym.acquire_net_contact_force_tensor(sim)
net_cf = gymtorch.wrap_tensor(_net_cf)

assert net_cf.shape[0] == num_bodies
net_cf = gymtorch.wrap_tensor(_net_cf).view(num_envs, num_bodies, 3)
franka_net_cf = net_cf[:, -num_franka_bodies:, :]

# net_cf_seq = torch.empty(0, 3).to(device)
# current pos and velocity
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)


def _store_observations(_dof_forces_seq, _franka_net_cf_seq):
    futils.cleanup_location(_force_store_dir, file_types=[".pt"])
    torch.save(_dof_forces_seq, _dof_seq_file)
    torch.save(_franka_net_cf_seq, _net_cf_file)

    consts = {'dof_stiffness': franka_dof_props["stiffness"],
              'dof_damping': franka_dof_props["damping"],
              'dof_friction': franka_dof_props["friction"]}

    with open(_tactile_const_file, 'wb') as file:
        pickle.dump(consts, file)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    raise ValueError("Max frame exceeded: Exiting")


# obstacle 1 & 2
cam_pos = gymapi.Vec3(0.2, -2.0, 1.0)
cam_target = gymapi.Vec3(0.2, 0.0, 1.0)

# obstacle 3 & 4
cam_pos = gymapi.Vec3(4.0, 0, 1.0)
cam_target = gymapi.Vec3(0, 0.0, 1.0)

gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# simulation loop
print(f"Executing SImulations .......")
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    frame_no = gym.get_frame_count(sim)
    # print(f"frame_no: {frame_no}")

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)
    gym.refresh_dof_force_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)

    # these are the current box and hand positions which gets refreshed.
    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.11 if controller == "ik" else 0.10

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]

    # if the box is gripped or not.
    gripped = (gripper_sep < 0.045) & (box_dist < grasp_offset + 0.5 * box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)

    # hand_restart is True if it was previously set to TRUE (after box drop) and distance to default position is > 0.02
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)

    # set return_to_start = True when box is gripped.
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

    # compute goal position and orientation
    # where operation is res = if x> y ? then a else b.
    # so goal_pos=init_pos if return_to_start is True else move to grasp_pos
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
    # Deploy control based on type
    if controller == "ik":
        ctrl = control_ik(dpose)
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + ctrl

    else:  # osc
        print("Warning: using osc/torque control")
        effort_action[:, :7] = control_osc(dpose)

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset + 0.02) | gripped

    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    # if box is above something by 0.6m set hand_restart = True (drop the box by opening fingers & going to default pos)
    hand_restart = hand_restart | (box_pos[:, 2] > 0.6)
    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * num_envs).to(device),
                            torch.Tensor([[0.04, 0.04]] * num_envs).to(device))
    pos_action[:, 7:9] = grip_acts

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    dof_forces_seq = torch.cat((dof_forces_seq, dof_forces.unsqueeze(0)), dim=0)
    assert num_envs == 1, "cat below works only for 1 env, otherwise the sequence will not make sense"
    franka_net_cf_seq = torch.cat((franka_net_cf_seq, franka_net_cf), dim=0)
    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

    # Store joint torques in file for observation.
    if frame_no >= 300:
        _store_observations(dof_forces_seq, franka_net_cf_seq)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
