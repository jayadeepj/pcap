import xml.etree.ElementTree as ET
import copy
import lxml.etree as etree

from anytree import RenderTree
from collections import deque
from math import pi


def gen_urdf(graph, outfile):
    for pre, fill, node in RenderTree(graph):
        tree_str = u"%s%s" % (pre, node.name)
        print(tree_str.ljust(8), node.length, node.radius, node.idx)

    outline, joint_template, link_template, leaf_template = gen_outline()
    mass_proportion = {}
    min_leaf_force = []

    traverse(graph,
             add_child,
             joint_template=joint_template,
             link_template=link_template,
             leaf_template=leaf_template,
             partial=outline,
             mass_proportion=mass_proportion,
             min_leaf_force=min_leaf_force)

    write_urdf(outline, f"../urdf/{outfile}")


def traverse(root, fn, **kwargs):
    """ Traverse via Breadth First Search"""
    # Initializing a queue
    queue = deque()
    queue.append(root)

    while len(queue) != 0:
        node = queue.popleft()
        fn(node, kwargs)
        for child in node.children:
            queue.append(child)


def add_child(node, kwargs):
    joint = add_joint(node, kwargs['joint_template'], kwargs['mass_proportion'], kwargs['min_leaf_force'])
    link = add_link(node, kwargs['link_template'])
    partial = kwargs['partial']

    component_comment = ET.Comment(text=f" Joint & Link: {node.name} Branch Id:{node.idx} Level:"
                                        f" {node.level}, Child idx:{node.child_idx} )")
    component_comment.tail = "\n"
    partial.append(component_comment)
    partial.append(joint)
    partial.append(link)


def add_joint(node, joint_template, mass_proportion, min_leaf_force):
    joint = copy.deepcopy(joint_template)
    if node.idx == 0:  # add trunk
        for joint_elem in joint.iter('joint'):
            joint_elem.set('name', f'joint-world-{node.name}')
            joint_elem.set('type', node.joint_type.name)  # always fixed for trunk
            search_remove_elem(joint_elem, './axis')
            search_remove_elem(joint_elem, './limit')
            search_remove_elem(joint_elem, 'dynamics')
            search_update_elem(joint_elem, 'parent', 'link', 'world')
            search_update_elem(joint_elem, 'child', 'link', f'link-{node.name}')
    else:
        for joint_elem in joint.iter('joint'):
            joint_name = f"joint-{node.parent.name}-to-{node.name}"
            joint_elem.set('name', joint_name)
            joint_elem.set('type', node.joint_type.name)

            # set relations
            search_update_elem(joint_elem, './parent', 'link', f'link-{node.parent.name}')
            search_update_elem(joint_elem, './child', 'link', f'link-{node.name}')

            # set origin
            joint_origin_xyz, joint_origin_rpy, joint_rotation_axis = calc_joint_origin(node)
            search_update_elem(joint_elem, './origin', 'xyz', joint_origin_xyz)
            search_update_elem(joint_elem, './origin', 'rpy', joint_origin_rpy)
            search_update_elem(joint_elem, './axis', 'xyz', joint_rotation_axis)

            # setting joint limit to 20
            limit_lower = -1 * round(pi / 9, 3)
            limit_upper = 1 * round(pi / 9, 3)
            search_update_elem(joint_elem, './limit', 'lower', f'{limit_lower}')
            search_update_elem(joint_elem, './limit', 'upper', f'{limit_upper}')

            # set friction and damping
            search_update_elem(joint_elem, './dynamics', 'damping', f'{node.damping}')
            search_update_elem(joint_elem, './dynamics', 'friction', f'{node.friction}')

    return joint


def calc_joint_origin(node):
    joint_origin_z = round((node.parent.length - 0.01), 3)

    branch_angle_rpy = node.branch_angle_rpy
    branch_rotation_axis = node.branch_rotation_axis

    joint_origin_xyz = f'0 0 {joint_origin_z}'
    joint_origin_rpy = f'{branch_angle_rpy[0]} {branch_angle_rpy[1]} {branch_angle_rpy[2]}'
    joint_rotation_axis = f'{branch_rotation_axis[0]} {branch_rotation_axis[1]} {branch_rotation_axis[2]}'

    return joint_origin_xyz, joint_origin_rpy, joint_rotation_axis


def add_link(node, link_template):
    link = copy.deepcopy(link_template)
    if node.idx == 0:  # add trunk
        for link_elem in link.iter('link'):
            link_elem.set('name', f'link-{node.name}')
            search_update_elem(link_elem, './visual/geometry/cylinder', 'length', f'{node.length}')
            search_update_elem(link_elem, './visual/geometry/cylinder', 'radius', f'{node.radius}')
            search_update_elem(link_elem, './visual/origin', 'xyz', f'0 0 {round((node.length / 2), 3)}')
            material_name, material_rgba = gen_material(node)
            search_update_elem(link_elem, './visual/material', 'name', f'{material_name}')
            search_update_elem(link_elem, './visual/material/color', 'rgba', f'{material_rgba}')

            search_update_elem(link_elem, './collision/geometry/cylinder', 'length', f'{node.length}')
            search_update_elem(link_elem, './collision/geometry/cylinder', 'radius', f'{node.radius}')
            search_update_elem(link_elem, './collision/origin', 'xyz', f'0 0 {round((node.length / 2), 3)}')

            search_update_elem(link_elem, './inertial/mass', 'value', f'{node.mass}')
            search_update_elem(link_elem, './inertial/origin', 'xyz', f'0 0 {round((node.length / 2), 3)}')

    else:
        for link_elem in link.iter('link'):
            link_elem.set('name', f'link-{node.name}')
            search_update_elem(link_elem, './visual/geometry/cylinder', 'length', f'{node.length}')
            search_update_elem(link_elem, './visual/geometry/cylinder', 'radius', f'{node.radius}')
            search_update_elem(link_elem, './visual/origin', 'xyz', f'0 0 {round((node.length / 2), 3)}')

            material_name, material_rgba = gen_material(node)
            search_update_elem(link_elem, './visual/material', 'name', f'{material_name}')
            search_update_elem(link_elem, './visual/material/color', 'rgba', f'{material_rgba}')

            search_update_elem(link_elem, './collision/geometry/cylinder', 'length', f'{node.length}')
            search_update_elem(link_elem, './collision/geometry/cylinder', 'radius', f'{node.radius}')
            search_update_elem(link_elem, './collision/origin', 'xyz', f'0 0 {round((node.length / 2), 3)}')

            search_update_elem(link_elem, './inertial/mass', 'value', f'{node.mass}')
            search_update_elem(link_elem, './inertial/origin', 'xyz', f'0 0 {round((node.length / 2), 3)}')
            search_update_elem(link_elem, './inertial/inertia', 'ixx', f'{node.moi.xx}')
            search_update_elem(link_elem, './inertial/inertia', 'ixy', f'{node.moi.xy}')
            search_update_elem(link_elem, './inertial/inertia', 'ixz', f'{node.moi.xz}')
            search_update_elem(link_elem, './inertial/inertia', 'iyy', f'{node.moi.yy}')
            search_update_elem(link_elem, './inertial/inertia', 'iyz', f'{node.moi.yz}')
            search_update_elem(link_elem, './inertial/inertia', 'izz', f'{node.moi.zz}')

    return link


def gen_material(node):
    material_rgba = ' '.join([str(c) for c in node.material_rgba])
    return node.material_name, material_rgba


def search_remove_elem(parent, elem_name):
    for removable in parent.findall(elem_name):
        parent.remove(removable)


def search_update_elem(parent, elem_name, elem_key, elem_val):
    for elem in parent.findall(elem_name):
        elem.set(elem_key, elem_val)


def gen_outline():
    template = ET.parse('../urdf/tree/gen/tree_template.urdf')
    root = template.getroot()

    joint_template, link_template, leaf_template = None, None, None
    for link in root.findall('link'):
        name = link.get('name')
        if name == "default_link_name":
            link_template = copy.deepcopy(link)
            root.remove(link)

    for joint in root.findall('joint'):
        name = joint.get('name')
        if name == "default_joint_name":
            joint_template = copy.deepcopy(joint)
            root.remove(joint)

    for link in root.findall('link'):
        name = link.get('name')
        if name == "default_leaf_name":
            leaf_template = copy.deepcopy(link)
            root.remove(link)

    return root, joint_template, link_template, leaf_template


def print_struct(element):
    print(convert_to_str(element))


def convert_to_str(element):
    content = ET.tostring(element)
    xml = etree.XML(content)
    xml_str = etree.tostring(xml, pretty_print=True, encoding="unicode")
    # remove empty lines
    return '\n'.join([line for line in xml_str.split('\n') if line.strip() != ""])


def write_urdf(element, path):
    xml_str = convert_to_str(element)
    with open(path, "w") as f:
        f.write(xml_str)
