""" File to test TurtleState => TurtleBranch => TreeBranch pipeline"""

from common.helpers.domains import LShape, TreeMetaAttrs
from common.helpers import futils
from simulation.lsystems1.three.fractal import turtle
from simulation.lsystems1.three.assemblers import tree_generator, urdf_generator
from anytree import RenderTree
import simulation.lsystems1.three.conf.yaml_parser as parser
import os
import copy
import pickle
import simulation
import random

random.seed(42)

parser.yaml_file = f'{os.path.dirname(simulation.lsystems1.three.conf.__file__)}/randomised/ternary_a.yaml'


def find_dof_roots(_t_root, _tree_config, _l_configs):
    """ Pre generate the tree to find the right place for the possible dof root.
    Note: This code is garbage, since the recursion is re applied, but it works. """
    mock_trunk = tree_generator \
        .turtle_branch_to_tree(t_root=_t_root,
                               t_config=_tree_config,
                               l_config=_l_configs)

    _dof_root_cnts = []  # list of possible dof roots (links) that has between 80 & 150 branches.
    for m_pre, _, m_node in RenderTree(mock_trunk):
        if 50 <= len(m_node.descendants) <= 150:  # gym can handle only 256 dofs
            _dof_root_cnts.append((m_node.name, len(m_node.descendants)))

    _dof_cnts = [t[1] for t in _dof_root_cnts]

    # Check if all second elements are equal
    assert all(x == _dof_cnts[0] for x in _dof_cnts), f"All dof counts from root must be equal:'.{_dof_cnts}"

    _dof_root_options = [t[0] for t in _dof_root_cnts]

    tree_generator.TreeBranch.unique = -1  # reset index to trunk
    return _dof_root_options


# Run the entire urdf generation pipeline
if __name__ == '__main__':
    user_input = input("Do you want to continue? (yes/no): ").strip().lower()
    if user_input not in ('yes', 'no'):
        print("Please enter 'yes' or 'no'.")

    if user_input != 'yes':
        quit()

    lsystem_config_base = parser.yaml_to_lsystem()
    tree_config_base = parser.yaml_to_tree_config()
    outfile_base = parser.out_file()
    l_strings, l_configs = tree_generator.yaml_to_l_string(l_config=lsystem_config_base)

    # cleanup paths
    out_file_seq_dir = os.path.abspath(os.path.dirname(f"../urdf/{outfile_base}"))
    outfile_raw_dir = os.path.dirname(out_file_seq_dir)

    outfile_root_parent = os.path.dirname(outfile_raw_dir)
    all_par_folders = [os.path.join(outfile_root_parent, _f) for _f in os.listdir(outfile_root_parent) if
                       os.path.isdir(os.path.join(outfile_root_parent, _f))]

    # delete all parallel folders so that the train is also deleted with raw.
    for to_clean in all_par_folders:
        if ('raw' in to_clean or 'train' in to_clean) and 'raw_test' not in to_clean:
            futils.cleanup_sub_folders_n_files(to_clean, folder_prefix='ta_', file_types=(".pkl", ".npy"))

    for r_idx, l_string in enumerate(l_strings):
        tree_generator.TreeBranch.unique = -1

        tree_config = copy.deepcopy(tree_config_base)
        outfile = outfile_base.format(r_idx=r_idx) if 'r_idx' in outfile_base else outfile_base

        tree_generator \
            .write_to_file(string=l_string,
                           out_file=f"../urdf/{outfile.format(r_idx=r_idx)}".replace('.urdf', f'_l_string.txt'))

        print('l_string >>', l_string)
        turtle_lines = turtle \
            .l_string_to_turtle_lines(l_string=l_string,
                                      l_config=l_configs[r_idx])
        t_root = turtle.turtle_lines_to_branches(turtle_lines)

        turtle_graph = ""
        for pre, fill, node in RenderTree(t_root):
            node_str = f"{pre}-{node.t_name()}"
            turtle_graph += node_str + "\n"

        tree_generator \
            .write_to_file(string=turtle_graph,
                           out_file=f"../urdf/{outfile}".replace('.urdf', '_t_graph.txt'))

        print('\n')

        if tree_config.dof_root is None or 'auto':
            dof_root_options = find_dof_roots(t_root, tree_config, l_configs[r_idx])
            tree_config.dof_root = random.choice(dof_root_options)

        trunk = tree_generator \
            .turtle_branch_to_tree(t_root=t_root,
                                   t_config=tree_config,
                                   l_config=l_configs[r_idx])

        print('\n')

        l_shape_dict = {}
        for pre, fill, node in RenderTree(trunk):
            tree_str = u"%s%s" % (pre, node.name)
            print(tree_str.ljust(8), node.length, node.radius, node.idx, node.branch_angle_rpy, len(node.descendants))
            l_shape_dict[node.name] = LShape(node.length, node.radius)

        tree_meta_attrs_fp = f"../urdf/{outfile}".replace('.urdf', '_meta.pkl')
        tree_meta_attrs = TreeMetaAttrs(urdf_path=outfile, l_shape_dict=l_shape_dict, dof_root=tree_config.dof_root)
        with open(tree_meta_attrs_fp, 'wb') as file:
            pickle.dump(tree_meta_attrs, file)

        urdf_generator.gen_urdf(trunk, outfile)
