import math
import numpy as np
import simulation.basic.conf.yaml_parser as parser


def gen_branch_lengths(trunk_len, max_level):
    """ Returns a set of progressively reducing branch lengths
     for a given trunk length & max depth. Level 0 implies trunk itself.
     Computed based on fractal canopy structure"""

    # Every subsequent tree branch will vary in size from
    # current_size * (reduction_factor  drawn from a normal distribution with params below)
    len_reduction_factor_mean = parser.config()['geometry']['len_reduction_factor_mean']
    len_reduction_factor_sd = parser.config()['geometry']['len_reduction_factor_sd']
    include_dummy_level = parser.config()['dummy_level']['include']
    dummy_branch_len_red_factor = parser.config()['dummy_level']['branch_len_red_factor']
    randomisation = parser.config()['randomisation']

    if randomisation:
        print("Original trunk_len without randomisation:", trunk_len)
        trunk_len = np.random.normal(loc=trunk_len,
                                     size=1,
                                     scale=0.025)[0]
        print("trunk_len after randomisation:", trunk_len)
        assert trunk_len > 0.0

    branch_lengths = [trunk_len]

    if max_level == 0:
        return branch_lengths

    for i in range(1, max_level):
        if randomisation:
            rand_proportion = np.random.normal(loc=len_reduction_factor_mean,
                                               size=1,
                                               scale=len_reduction_factor_sd)[0]
        else:
            rand_proportion = len_reduction_factor_mean

        trunk_len = trunk_len * rand_proportion
        branch_lengths += [trunk_len]

    # reduce the length of the last branch by a factor

    if include_dummy_level:
        branch_lengths[-1] = branch_lengths[-1] * dummy_branch_len_red_factor

    return [round(l, 3) for l in branch_lengths]


def gen_branch_radius(trunk_rad, max_level, branch_per_level=2):
    """ Returns a set of progressively reducing branch radius
     for a given trunk radius & max depth. Level 0 implies trunk itself.
     Computed based on Da Vinci cross-section area preservation rule"""

    branch_radii = [trunk_rad]

    if max_level == 0:
        return branch_radii

    # TBD: for now assuming equal child branch diameter, needs to be randomised
    for i in range(1, max_level):
        child_branch_radius = branch_radii[i - 1] / math.sqrt(branch_per_level)
        branch_radii += [child_branch_radius]

    return [round(l, 3) for l in branch_radii]


def gen_child_cnt(max_level, max_child_count):
    """ The number of children at each level a random number between [1, max_child_count] """

    include_dummy_level = parser.config()['dummy_level']['include']
    randomise_child_cnt = parser.config()['randomise_child_cnt']
    tail = [1] + [0] if include_dummy_level else [0]
    true_level = max_level - 1 if include_dummy_level else max_level

    if randomise_child_cnt:
        return [max_child_count] + np.random.randint(1, max_child_count + 1, size=(true_level - 2)).tolist() + tail

    return [max_child_count for _ in range(1, true_level)] + tail


def tree_params(trunk_len, trunk_rad, max_level, max_child_count):
    branch_radii = gen_branch_radius(trunk_rad=trunk_rad, max_level=max_level, branch_per_level=2)
    branch_lengths = gen_branch_lengths(trunk_len=trunk_len, max_level=max_level)

    geometry_template = """<cylinder length="{branch_len}" radius="{branch_rad}"/>"""

    for branch_len, branch_rad in zip(branch_lengths, branch_radii):
        entry = geometry_template.format(branch_len=branch_len, branch_rad=branch_rad)
        print(entry, "\n")


def euclidean_dist(point1, point2):
    """ Given 2 [x,y,z] coordinates, return the dist"""
    point1_np = np.array(point1)
    point2_np = np.array(point2)
    return np.linalg.norm(point2_np - point1_np)


def test_tree_param():
    tree_params(trunk_len=2, trunk_rad=0.2, max_level=5, max_child_count=4)
