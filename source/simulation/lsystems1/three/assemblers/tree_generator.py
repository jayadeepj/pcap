import os
from anytree import NodeMixin
from enum import Enum
import numpy as np
import copy
from simulation.lsystems1.three.fractal import rewriter
from simulation.lsystems1.three.fractal.turtle import TurtleBranch
from simulation.lsystems1.utils import physics, rotation


class Twig:
    pass


class JointType(str, Enum):
    revolute = 'revolute'
    fixed = 'fixed'


class TreeBranch(Twig, NodeMixin):
    unique = -1

    def __init__(self, length, radius, mass, moi, branch_angle_rpy, branch_rotation_axis,
                 child_idx, child_cnt, joint_type: JointType, parent=None):
        super(Twig, self).__init__()

        self.length = length
        self.radius = radius
        self.mass = mass
        self.moi = moi
        self.friction = 0.0  # not used
        self.damping = 0.0  # not used

        self.child_cnt = child_cnt

        self.parent = parent

        # if the joint from this branch to parent is fixed/revolute. Ignored for trunk.
        self.joint_type = joint_type

        # wood color for all branches except at the max level
        self.material_rgba = [0.1, 0.2, 0.2, 1] if self.joint_type == JointType.fixed else [0.1, 0.8, 0.2, 1]
        self.material_name = 'brown' if self.joint_type == JointType.fixed else 'green'

        # self.material_rgba = [0.201, 0.002, 0.071,1]
        # self.material_name = 'red'

        # mass of all children including this branch/total mass of tree at first non-fixed joint
        self.branch_angle_rpy = branch_angle_rpy
        self.branch_rotation_axis = branch_rotation_axis

        self.idx = self.__class__.unique + 1
        self.level = self.parent.level + 1 if parent is not None else 0
        self.name = f'branch-B{self.idx}L{self.level}P{parent.idx}' \
            if parent is not None else f'trunk-B{self.idx}L{self.level}'
        self.child_idx = child_idx
        self.__class__.unique += 1


def check_joint_type(tree_branch_parent, t_config):
    """ If the configured dof root matches the link name or if the immediate parent is revolute,
     set the current node as revolute else fixed. So, set all children of dof_root as revolute"""

    if tree_branch_parent.name == t_config.dof_root or tree_branch_parent.joint_type == JointType.revolute:
        return JointType.revolute

    return JointType.fixed


def turtle_branch_to_tree(t_root: TurtleBranch, l_config, t_config):
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
        node_joint_type = check_joint_type(tree_branch_parent, t_config)
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
    #assert train_cnt > 0 and (train_cnt & (train_cnt - 1)) == 0, " Train file count must be a power of 2, e.g. 16"

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
