import os
from collections import deque

import numpy as np
import copy
from simulation.lsystems1.three.fractal import rewriter
from simulation.lsystems1.three.fractal.turtle import TurtleBranch
from simulation.lsystems1.utils import physics, rotation
from simulation.lsystems2.three.assemblers.domain import *
import random

random.seed(42)


def check_joint_type(tree_branch_parent, t_config, child_cnt):
    """ If the configured dof root matches the link name or if the immediate parent is revolute,
     set the current node as revolute else fixed. So, set all children of dof_root as revolute"""

    if tree_branch_parent.name == t_config.flex_root or tree_branch_parent.joint_type == JointType.spherical:
        if child_cnt != 0:  # don't use spherical for the last level branch to reduce dofs.
            return JointType.spherical

    if tree_branch_parent.name == t_config.dof_root or tree_branch_parent.joint_type == JointType.revolute or tree_branch_parent.joint_type == JointType.spherical:
        return JointType.revolute

    return JointType.fixed


def turtle_branch_to_tree(t_root: TurtleBranch, t_config):
    trunk = gen_branch_graph(t_root, t_config)
    add_leaf_n_fruit_nodes(trunk, t_config)
    return trunk


def add_leaf_n_fruit_nodes(root, t_config):
    """ Traverse & add leaves, fruits via Breadth First Search"""

    print("Adding Fruits/Leaves ..... ")
    queue = deque()
    queue.append(root)

    while len(queue) != 0:
        node = queue.popleft()
        if 'branch' in node.name:
            if t_config.include_fruits:
                add_fruits(node, t_config)
                raise ValueError("Mass/Inertia Changes for flexible links yet to be implemented in fruit template.")

            if t_config.include_leaves:
                raise ValueError("Not implemented yet.")

        for child in node.children:
            if not isinstance(child, LeafConn):
                if not isinstance(child, FruitConn):
                    queue.append(child)


def add_fruits(branch_node, t_config):
    # add fruits only in the selected flexible branches.
    if t_config.fruit_branches is None or branch_node.name not in t_config.fruit_branches:
        return
    if 'branch' not in branch_node.name:
        return

    fruits = []

    # if you randomise, no of links in the body will change b/w envs causing rb_states to have varying shapes.
    # fruits_cnt = np.random.randint(1, t_config.fruits_cnt + 1)  # np low is inclusive, but hi is exclusive
    fruits_cnt = t_config.fruits_cnt

    fruit_locs_z = sample_fruit_locs(length=round((branch_node.length - 0.01), 3), num_samples=fruits_cnt)
    for child_idx in range(fruits_cnt):
        fruit_origin_xyz, fruit_angle_rpy = calc_fruit_origin_xyz_rpy(fruit_locs_z[child_idx])
        fc = FruitConn(fruit_origin_xyz, fruit_angle_rpy, child_idx, parent=branch_node)
        fruits.append(fc)

    branch_node.children = tuple(fruits) if branch_node.children is None else branch_node.children + tuple(fruits)


def sample_fruit_locs(length, num_samples):
    """ Uniformly select samples from the first and the third slice of the branch"""

    if length <= 0 or num_samples <= 0:
        raise ValueError(f"Length and number of samples must be positive {length}, {num_samples}")

    segment_length = length / 3

    samples = []
    for _ in range(num_samples):
        segment = random.choice([0, 2])  # 0 for first segment, 2 for third segment
        point = random.uniform(0, segment_length) if segment == 0 else random.uniform(2 * segment_length, length)
        samples.append(point)

    return samples


def random_xy_tuple(xy_opts):
    x, y = random.sample(xy_opts, 2)
    x *= random.choice([-1, 1])
    y *= random.choice([-1, 1])
    return (x, y) if random.choice([True, False]) else (y, x)


def calc_fruit_origin_xyz_rpy(fruit_loc_z):
    fruit_angle_rpy = (0., 0., 0.)
    fruit_origin_xy = random_xy_tuple([0.03, 0.01])
    fruit_origin_xyz = (fruit_origin_xy[0], fruit_origin_xy[1], fruit_loc_z)
    return fruit_origin_xyz, fruit_angle_rpy


def gen_branch_graph(t_root: TurtleBranch, t_config):
    """TurtleBranch(s) => TreeBranch(s) """

    def _length(_node):
        start = _node.turtle_line.start
        end = _node.turtle_line.end
        return start.distance_to(end)

    def _radius(_node):
        # use the parameterised by lsystem to generate the _radius, computed as 1/2 of width
        return _node.turtle_line.width / 2.0

    # turtle_trunk is an instance of TurtleBranch. trunk is an instance of TreeBranch.
    # turtle_branch is an instance of TurtleBranch. branch is an instance of TreeBranch
    turtle_trunk = t_root

    t_length = _length(turtle_trunk) * t_config.len_scale_factor
    t_radius = _radius(turtle_trunk) * t_config.rad_scale_factor

    t_mass = physics.mass(height=t_length, radius=t_radius, density=t_config.density)
    t_moi = physics.moment_of_inertia_cylinder(height=t_length, radius=t_radius, density=t_config.density)

    trunk = TreeBranch(
        length=t_length,
        radius=t_radius,
        mass=t_mass,
        moi=t_moi,
        # ignore angle & rotation for trunk
        branch_angle_rpy=(0, 0, 0),
        branch_rotation_axis='0 0 0',
        child_idx=0,
        parent=None,
        child_cnt=len(turtle_trunk.children),
        # fixed joint for trunk
        joint_type=JointType.fixed)

    # a dictionary that stores the TurtleBranch to TreeNode relation to assign the parents correctly
    branch_store = {turtle_trunk: trunk}

    for turtle_branch in turtle_trunk.descendants:
        b_length = _length(turtle_branch) * t_config.len_scale_factor
        b_radius = _radius(turtle_branch) * t_config.rad_scale_factor
        b_mass = physics.mass(b_length, b_radius, density=t_config.density)
        b_moi = physics.moment_of_inertia_cylinder(height=b_length, radius=b_radius, density=t_config.density)
        turtle_branch_parent = turtle_branch.parent

        # TODO: fix rotation axis
        b_rotation_axis = (0, 1, 0)

        # pass the start and end points of the branch and its parent to compute the rpy in between
        b_angle_rpy = rotation.calculate_rpy(l1_start=turtle_branch_parent.turtle_line.start,
                                             l1_end=turtle_branch_parent.turtle_line.end,
                                             l2_start=turtle_branch.turtle_line.start,
                                             l2_end=turtle_branch.turtle_line.end)

        tree_branch_parent = branch_store.get(turtle_branch_parent)
        node_joint_type = check_joint_type(tree_branch_parent, t_config, child_cnt=len(turtle_branch.children))
        branch = TreeBranch(
            length=b_length,
            radius=b_radius,
            mass=b_mass,
            moi=b_moi,
            branch_angle_rpy=b_angle_rpy,
            branch_rotation_axis=b_rotation_axis,
            child_idx=0,
            parent=tree_branch_parent,
            child_cnt=len(turtle_branch.children),
            joint_type=node_joint_type)

        branch_store.update({turtle_branch: branch})

    return trunk


def write_to_file(string, out_file):
    try:
        # Extract the directory path from the output file
        dir_path = os.path.dirname(out_file)

        # Create the directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with open(out_file, 'w') as file:
            file.write(string)
        print(f"Successfully wrote the string to {out_file}.")
    except IOError:
        print(f"Error: Unable to write to {out_file}.")


def yaml_to_l_string(l_config):
    l_configs = randomise_config(l_config)
    for l_config in l_configs:
        print(l_config.free_params)

    nl_strings = [rewriter.gen_lsystem(l_config)[-1] for l_config in l_configs]
    return nl_strings, l_configs


def randomise_config(l_config):
    if not l_config.randomise:
        return [l_config]

    rand_l_configs = []

    assert l_config.randomise_cnt > 0, "Invalid randomise count, at least 1 necessary for the default params"

    train_cnt = l_config.randomise_cnt
    # assert train_cnt > 0 and (train_cnt & (train_cnt - 1)) == 0, " Train file count must be a power of 2, e.g. 16"

    if l_config.randomise_cnt == 1:  # no randomisation, just use the config as is
        rand_l_configs.append(l_config)
        return rand_l_configs

    for _ in range(l_config.randomise_cnt):
        l_clone = copy.deepcopy(l_config)
        rand_params = {}
        for p_key, p_value in l_config.free_params.items():
            abs_std = l_config.rel_std * abs(p_value)  # Calculate absolute std based on the magnitude
            rand_value = round(p_value + np.random.normal(0, abs_std), 2)
            rand_params[p_key] = rand_value

        # Add modified_params to the list
        l_clone.free_params = rand_params

        abs_e_std = l_config.rel_std * abs(l_config.e)
        l_clone.e = round(l_config.e + np.random.normal(0, abs_e_std), 2)

        abs_sigma_std = l_config.rel_std * abs(l_config.sigma)
        l_clone.sigma = round(l_config.sigma + np.random.normal(0, abs_sigma_std), 2)

        abs_theta_std = l_config.rel_std * abs(l_config.theta)
        l_clone.theta = round(l_config.theta + np.random.normal(0, abs_theta_std), 2)

        # l_clone.n= random.choice([6,7,8])

        rand_l_configs.append(l_clone)

    return rand_l_configs
