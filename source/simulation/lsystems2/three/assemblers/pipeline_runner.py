""" File to test TurtleState => TurtleBranch => TreeBranch pipeline"""
import functools
import os
import copy
import pickle
import simulation
import random
from anytree import RenderTree
from concurrent.futures import ProcessPoolExecutor

from common.helpers.domains import LShape, TreeMetaAttrs
from common.helpers import futils
from simulation.lsystems1.three.fractal import turtle
from simulation.lsystems2.three.assemblers import tree_generator, urdf_generator

import simulation.lsystems2.three.conf.yaml_parser as parser
from simulation.lsystems2.three.assemblers.domain import NodeType

random.seed(42)

parser.yaml_file = f'{os.path.dirname(simulation.lsystems2.three.conf.__file__)}/pliable04/ta/ternary_a.yaml'


def find_dof_roots(_t_root, _tree_config, _l_configs):
    """ Pre generate the tree to find the right place for the possible dof root.
    Note: This code is garbage, since the recursion is re applied, but it works. """
    mock_trunk = tree_generator \
        .turtle_branch_to_tree(t_root=_t_root,
                               t_config=_tree_config)

    _dof_root_cnts = []  # list of possible dof roots (links) that has between 80 & 150 branches.
    for m_pre, _, m_node in RenderTree(mock_trunk):
        if 40 <= len(m_node.descendants) <= 120:  # gym can handle only 256 dofs
            _dof_root_cnts.append((m_node, len(m_node.descendants)))

    _dof_root_cnt_vals = [t[1] for t in _dof_root_cnts]
    print(f"Root dof count option set:", {*_dof_root_cnt_vals})

    _root_option = random.choice([t for t in _dof_root_cnts])
    _dof_root_sel, _dof_root_sel_cnt = _root_option[0], _root_option[1]

    # Option 1: Choose any random child node of _dof_root_sel as the _dof_flex_root_sel
    #         : In this case the check_joint_type method to exclude descendants without children must be removed.
    # ===========================================================================================================
    # _dof_flex_cnts = []  # flexible/spherical joints
    #
    # for m_pre, _, m_node in RenderTree(_dof_root_sel):
    #     if 16 <= len(m_node.descendants) <= 59:  # gym can handle only 256 dofs
    #         _dof_flex_cnts.append((m_node, len(m_node.descendants)))
    #
    # _dof_flex_cnt_vals = [t[1] for t in _dof_flex_cnts]
    # print("Flexible dof count option set:", {*_dof_flex_cnt_vals})
    #
    # _flex_option = random.choice([t for t in _dof_flex_cnts])
    # _dof_flex_root_sel, _dof_flex_root_sel_cnt = _flex_option[0], _flex_option[1]

    # _total_dof_cnt = _dof_root_sel_cnt + 2 * _dof_flex_root_sel_cnt

    # ===========================================================================================================
    # Option 2: Keep _dof_flex_root_sel same as the _dof_root_sel
    #         : In this case the check_joint_type method to exclude descendants without children must be added.
    # ===========================================================================================================

    _dof_flex_root_sel, _dof_flex_root_sel_cnt = _root_option[0], _root_option[1]
    desc_parents_cnt = count_descendants_with_children(_dof_root_sel)

    # this number must be the same for all trees
    print(f"descendants_with_children for {_dof_root_sel}: {desc_parents_cnt}, Total:{len(_dof_root_sel.descendants)}")
    _total_dof_cnt = _dof_root_sel_cnt + 2 * desc_parents_cnt

    # ===========================================================================================================

    print(f"Selected Root: {_dof_root_sel.name}, Flex: {_dof_flex_root_sel.name}")
    print("Total dof count: ", _total_dof_cnt)  # assuming 3 spherical joints

    assert _total_dof_cnt <= 230, "Too Many dofs. Max 256 allowed including fruits"
    if _total_dof_cnt != 198:  # _total_dof_cnt can be any value as long it is same for all trees.
        raise ValueError("Total dofs must be the same. Replace with any number ")

    fruit_branch_opts = []
    # iterate through flexible branches to choose n branches to put the fu
    for m_pre, _, m_node in RenderTree(_dof_flex_root_sel):
        # if m_node.children is not None and len(m_node.children) != 0:  # don't consider branches that have no children
        fruit_branch_opts.append(m_node.name)

    _fruit_branches_sel = random.sample(fruit_branch_opts, k=1)  # choose only one branch for now

    tree_generator.TreeBranch.unique = -1  # reset index to trunk
    return _dof_root_sel.name, _dof_flex_root_sel.name, _fruit_branches_sel


def count_descendants_with_children(node):
    """
    Counts all descendants of the given node, excluding nodes without children.
    Args:  node (Node): The root anytree node for counting descendants.
    """
    count = 0
    for descendant in node.descendants:
        # Count only if the descendant has children
        if descendant.children:
            count += 1

    return count


def par_processor(r_idx, l_string, tree_config_base, outfile_base, l_configs):
    tree_generator.TreeBranch.unique = -1

    tree_config = copy.deepcopy(tree_config_base)
    outfile = outfile_base.format(r_idx=r_idx) if 'r_idx' in outfile_base else outfile_base

    tree_generator \
        .write_to_file(string=l_string,
                       out_file=f"../urdf/{outfile.format(r_idx=r_idx)}".replace('.urdf', f'_l_string.txt'))

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

    if tree_config.dof_root is None or 'auto':
        dof_root_sel, dof_flex_sel, fruit_branches_sel = find_dof_roots(t_root, tree_config, l_configs[r_idx])
        tree_config.dof_root = dof_root_sel
        tree_config.flex_root = dof_flex_sel
        tree_config.fruit_branches = fruit_branches_sel

    trunk = tree_generator \
        .turtle_branch_to_tree(t_root=t_root,
                               t_config=tree_config)

    l_shape_dict = {}
    for pre, fill, node in RenderTree(trunk):
        tree_str = u"%s%s" % (pre, node.name)
        # print(tree_str.ljust(8), node.idx, len(node.descendants))

        if NodeType.from_node(node) == NodeType.branch:  # skip leaves/fruits
            l_shape_dict[node.name] = LShape(node.length, node.radius)

    tree_meta_attrs_fp = f"../urdf/{outfile}".replace('.urdf', '_meta.pkl')
    tree_meta_attrs = TreeMetaAttrs(urdf_path=outfile, l_shape_dict=l_shape_dict, dof_root=tree_config.dof_root)
    with open(tree_meta_attrs_fp, 'wb') as file:
        pickle.dump(tree_meta_attrs, file)

    urdf_generator.gen_urdf(trunk, outfile, tree_config.spherical_dim)
    print("-----------------------------------------------------------------------------------------------------")


if __name__ == '__main__':

    user_input = input(f"Files will be overwritten. Do you want to continue? (yes/no): ").strip().lower()
    if user_input not in ('yes', 'no'):
        print("Please enter 'yes' or 'no'.")

    if user_input != 'yes':
        quit()

    _lsystem_config_base = parser.yaml_to_lsystem()
    _tree_config_base = parser.yaml_to_tree_config()
    _outfile_base = parser.out_file()
    l_strings, _l_configs = tree_generator.yaml_to_l_string(l_config=_lsystem_config_base)

    # cleanup paths
    out_file_seq_dir = os.path.abspath(os.path.dirname(f"../urdf/{_outfile_base}"))
    outfile_raw_dir = os.path.dirname(out_file_seq_dir)

    outfile_root_parent = os.path.dirname(outfile_raw_dir)


    def _ldir(_path):
        """ Last dir name"""
        return os.path.basename(os.path.normpath(_path))


    in_raw_dir_type = _ldir(outfile_raw_dir)  # raw or raw_test
    all_par_folders = [os.path.join(outfile_root_parent, _f) for _f in os.listdir(outfile_root_parent) if
                       os.path.isdir(os.path.join(outfile_root_parent, _f))]

    # delete all parallel folders so that the train is also deleted with raw.
    for to_clean in all_par_folders:
        to_clean_ldir = _ldir(to_clean)
        assert to_clean_ldir in ['raw_train', 'raw_test', 'temp', 'train', 'test'], f"Invalid dir {to_clean_ldir}"

        if to_clean_ldir in ['train' or 'test']:
            user_input = input(
                f"Final {to_clean_ldir} will be overwritten. Do you want to continue? (yes/no): ").strip().lower()
            if user_input not in ('yes', 'no'):
                print("Please enter 'yes' or 'no'.")

            if user_input != 'yes':
                quit()

    for to_clean in all_par_folders:
        to_clean_ldir = _ldir(to_clean)
        if (to_clean_ldir == 'raw' or to_clean_ldir == 'train') and in_raw_dir_type == 'raw':
            futils.cleanup_sub_folders_n_files(to_clean, folder_prefix='ta_', file_types=(".pkl", ".npy"))
        elif (to_clean_ldir == 'raw_test' or to_clean_ldir == 'test') and in_raw_dir_type == 'raw_test':
            futils.cleanup_sub_folders_n_files(to_clean, folder_prefix='ta_', file_types=(".pkl", ".npy"))
        elif to_clean_ldir == 'temp':
            futils.cleanup_sub_folders_n_files(to_clean, folder_prefix='ta_', file_types=(".pkl", ".npy"))

    # Create a partial function that includes static parameters
    partial_par_processor = functools.partial(par_processor, tree_config_base=_tree_config_base,
                                              outfile_base=_outfile_base, l_configs=_l_configs)

    with ProcessPoolExecutor(max_workers=6) as executor:
        # Submitting tasks to the pool for parallel execution
        results = executor.map(partial_par_processor, range(len(l_strings)), l_strings)

    # This check is required, because the exceptions from par processor are swallowed by ProcessPoolExecutor,
    futils.validate_file_types_in_sub_folders(folder_path=outfile_raw_dir, file_types=['.urdf'])
    print("Flexible Tree Generation Completed.")
