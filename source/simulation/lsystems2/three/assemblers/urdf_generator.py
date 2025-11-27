import xml.etree.ElementTree as ET
import copy
import lxml.etree as etree
from collections import namedtuple
from anytree import RenderTree
from collections import deque
from math import pi
from simulation.lsystems2.three.assemblers.domain import *
import numpy as np


def gen_urdf(graph, outfile, spherical_dim):
    for pre, fill, node in RenderTree(graph):
        tree_str = u"%s%s" % (pre, node.name)
        # print(tree_str.ljust(8), node.length, node.radius, node.idx)

    outline, templates = gen_outline(spherical_dim)
    mass_proportion = {}
    min_leaf_force = []

    traverse(graph,
             add_child,
             templates=templates,
             partial=outline,
             mass_proportion=mass_proportion,
             min_leaf_force=min_leaf_force,
             spherical_dim=spherical_dim)

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
    def _is_et_list(obj):
        return isinstance(obj, list) and all(isinstance(elem, ET.Element) for elem in obj)

    if NodeType.from_node(node) == NodeType.branch:  # skip leaves/fruits
        joints = add_joint(node, kwargs['templates'].joint, kwargs['templates'].spherical, kwargs['spherical_dim'])
        link = add_link(node, kwargs['templates'].link)
    elif NodeType.from_node(node) == NodeType.fruit:
        joints = add_fruit_joint(node, kwargs['templates'].fruit_joint)
        link = add_fruit_link(node, kwargs['templates'].fruit_link)
    else:
        raise ValueError("Only fruits and branches are implemented as of now.")

    partial = kwargs['partial']

    component_comment = ET.Comment(text=f" {node.name} Comp Id:{node.idx} Level:"
                                        f" {node.level}, Child idx:{node.child_idx} )")
    component_comment.tail = "\n"
    partial.append(component_comment)

    assert _is_et_list(joints) is True, "Only list of ET allowed for joints. Sp joint can include dummy links"
    for j_comp in joints:
        partial.append(j_comp)

    partial.append(link)


def add_fruit_joint(node, fruit_joint_template):
    if NodeType.from_node(node) == NodeType.fruit:
        fruit_joint_set = copy.deepcopy(fruit_joint_template)
        unique_x_link_name, unique_y_link_name = None, None
        for joint_elem in fruit_joint_set.iter('joint'):
            def_joint_name = joint_elem.get('name')
            if '-x-' in def_joint_name:
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.revolute.name)  # use revolute only
                search_update_elem(joint_elem, './parent', 'link', f'link-{node.parent.name}')

                fruit_origin_xyz = node.fruit_origin_xyz
                fruit_angle_rpy = node.fruit_angle_rpy

                fruit_joint_origin_xyz = f'{fruit_origin_xyz[0]} {fruit_origin_xyz[1]} {fruit_origin_xyz[2]}'
                fruit_joint_origin_rpy = f'{fruit_angle_rpy[0]} {fruit_angle_rpy[1]} {fruit_angle_rpy[2]}'

                search_update_elem(joint_elem, './origin', 'xyz', fruit_joint_origin_xyz)
                search_update_elem(joint_elem, './origin', 'rpy', fruit_joint_origin_rpy)

                def_x_link_name = search_elem(joint_elem, './child', 'link')
                unique_x_link_name = f'{def_x_link_name}-{node.idx}'
                search_update_elem(joint_elem, './child', 'link', unique_x_link_name)

            elif '-y-' in def_joint_name:
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.revolute.name)  # use revolute only

                search_update_elem(joint_elem, './parent', 'link', f'{unique_x_link_name}')
                def_y_link_name = search_elem(joint_elem, './child', 'link')
                unique_y_link_name = f'{def_y_link_name}-{node.idx}'

                search_update_elem(joint_elem, './child', 'link', unique_y_link_name)

            elif '-prismatic-' in def_joint_name:
                # ideally this joint should be in world z-axis, but for now fruit z-axis default.
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.prismatic.name)  # use revolute only

                search_update_elem(joint_elem, './parent', 'link', unique_y_link_name)
                search_update_elem(joint_elem, './child', 'link', f'link-{node.name}')

        for link_elem in fruit_joint_set.iter('link'):
            def_link_name = link_elem.get('name')
            if '-x-' in def_link_name:
                link_elem.set('name', unique_x_link_name)
            if '-y-' in def_link_name:
                link_elem.set('name', unique_y_link_name)

        return list(fruit_joint_set)  # return only the children


def add_joint(node, joint_template, spherical_template, spherical_dim):
    if node.idx == 0:  # add trunk
        basic_joint = copy.deepcopy(joint_template)
        for joint_elem in basic_joint.iter('joint'):
            joint_elem.set('name', f'joint-world-{node.name}')
            joint_elem.set('type', node.joint_type.name)  # always fixed for trunk
            search_remove_elem(joint_elem, './axis')
            search_remove_elem(joint_elem, './limit')
            search_remove_elem(joint_elem, 'dynamics')
            search_update_elem(joint_elem, 'parent', 'link', 'world')
            search_update_elem(joint_elem, 'child', 'link', f'link-{node.name}')
        return [basic_joint]
    elif node.joint_type == JointType.spherical:
        spherical_joint_set = copy.deepcopy(spherical_template)
        unique_x_link_name, unique_y_link_name = None, None
        for joint_elem in spherical_joint_set.iter('joint'):
            def_joint_name = joint_elem.get('name')
            if '-x-' in def_joint_name:
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.revolute.name)  # use revolute only
                search_update_elem(joint_elem, './parent', 'link', f'link-{node.parent.name}')

                joint_origin_xyz, joint_origin_rpy, joint_rotation_axis = calc_joint_origin(node)
                search_update_elem(joint_elem, './origin', 'xyz', joint_origin_xyz)
                search_update_elem(joint_elem, './origin', 'rpy', joint_origin_rpy)

                def_x_link_name = search_elem(joint_elem, './child', 'link')
                unique_x_link_name = f'{def_x_link_name}-{node.idx}'
                search_update_elem(joint_elem, './child', 'link', unique_x_link_name)

            elif '-y-' in def_joint_name and spherical_dim == 3:
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.revolute.name)  # use revolute only

                search_update_elem(joint_elem, './parent', 'link', f'{unique_x_link_name}')
                def_y_link_name = search_elem(joint_elem, './child', 'link')
                unique_y_link_name = f'{def_y_link_name}-{node.idx}'

                search_update_elem(joint_elem, './child', 'link', unique_y_link_name)

            elif '-y-' in def_joint_name and spherical_dim == 2:
                raise ValueError("If spherical_dim==2, there is no second dummy link")

            elif '-z-' in def_joint_name and spherical_dim == 3:
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.revolute.name)  # use revolute only

                search_update_elem(joint_elem, './parent', 'link', unique_y_link_name)
                search_update_elem(joint_elem, './child', 'link', f'link-{node.name}')

            elif '-z-' in def_joint_name and spherical_dim == 2:
                joint_name = f"{def_joint_name}-{node.parent.name}-to-{node.name}"
                joint_elem.set('name', joint_name)
                joint_elem.set('type', JointType.revolute.name)  # use revolute only

                search_update_elem(joint_elem, './parent', 'link', unique_x_link_name)
                search_update_elem(joint_elem, './child', 'link', f'link-{node.name}')

        for link_elem in spherical_joint_set.iter('link'):
            def_link_name = link_elem.get('name')
            if '-x-' in def_link_name:
                link_elem.set('name', unique_x_link_name)
            if '-y-' in def_link_name:
                link_elem.set('name', unique_y_link_name)

        return list(spherical_joint_set)  # return only the children
    else:
        basic_joint = copy.deepcopy(joint_template)
        for joint_elem in basic_joint.iter('joint'):
            joint_name = f"joint-{node.parent.name}-to-{node.name}"
            joint_elem.set('name', joint_name)
            joint_elem.set('type', node.joint_type.name)  # use fixed/revolute

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
        return [basic_joint]


def point_in_direction(rpy, dist):
    """ Given an rpy, find a point {dist} away from (0, 0, 0) Essentially find a mock origin."""
    assert dist <= 0.01, "More distance will disrupt the simulation appearence"
    roll, pitch, yaw = rpy

    # Convert roll, pitch, yaw to a direction vector using spherical coordinates
    # The yaw (z-axis rotation) determines the direction in the x-y plane
    # The pitch (y-axis rotation) determines the elevation angle
    # The roll isn't needed for calculating the direction; it represents rotation around the z-axis
    x = dist * np.cos(pitch) * np.cos(yaw)
    y = dist * np.cos(pitch) * np.sin(yaw)
    z = dist * np.sin(pitch)

    return (x, y, z)


def calc_joint_origin(node, shift_origin=False):
    joint_origin_z = round((node.parent.length), 5)

    branch_angle_rpy = node.branch_angle_rpy
    mock_origin = (0., 0., 0.)

    if shift_origin:
        # add a small distance away from the row origin in the direction of rpy,
        # to prevent self spherical dummy links coinciding.
        # Note: This messes up the appearance. change dist to 0
        mock_origin = point_in_direction(rpy=branch_angle_rpy, dist=0.01)

    branch_rotation_axis = node.branch_rotation_axis

    joint_origin_xyz = f'{mock_origin[0]} {mock_origin[1]} {mock_origin[2] + joint_origin_z}'
    joint_origin_rpy = f'{branch_angle_rpy[0]} {branch_angle_rpy[1]} {branch_angle_rpy[2]}'
    joint_rotation_axis = f'{branch_rotation_axis[0]} {branch_rotation_axis[1]} {branch_rotation_axis[2]}'

    return joint_origin_xyz, joint_origin_rpy, joint_rotation_axis


def add_fruit_link(node, fruit_link_template):
    link = copy.deepcopy(fruit_link_template)

    if NodeType.from_node(node) == NodeType.fruit:
        for link_elem in link.iter('link'):
            link_elem.set('name', f'link-{node.name}')
            search_update_elem(link_elem, './collision/geometry/sphere', 'radius', f'{node.radius}')

    return link


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


def search_elem(parent, elem_name, elem_key):
    for elem in parent.findall(elem_name):
        def_elem_val = elem.get(elem_key)
        if def_elem_val is not None and len(def_elem_val.strip()) != 0:
            return def_elem_val.strip()


def gen_outline(spherical_dim):
    outline, joint_template, link_template, leaf_template = gen_basic_outline()
    spherical_template = gen_spherical_outline(spherical_dim)
    fruit_joint_template, fruit_link_template = gen_fruit_outline()

    # Create a namedtuple called Templates
    Templates = namedtuple('Templates', ['joint', 'link', 'leaf', 'spherical', 'fruit_joint', 'fruit_link'])

    # Create an instance of Templates with all templates except outline
    templates = Templates(joint_template, link_template, leaf_template, spherical_template,
                          fruit_joint_template, fruit_link_template)

    return outline, templates


def gen_basic_outline():
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


def gen_spherical_outline(spherical_dim):
    template = ET.parse(f'../urdf/tree/gen/spherical_joint_template-{spherical_dim}d.urdf')
    root = template.getroot()

    spherical_template = None

    for joint in root.findall('spherical_joint_set'):
        name = joint.get('name')
        if name == "spherical_joint":
            spherical_template = copy.deepcopy(joint)
            root.remove(joint)

    return spherical_template


def gen_fruit_outline():
    template = ET.parse('../urdf/tree/gen/fruit_template.urdf')
    root = template.getroot()

    fruit_joint_template, fruit_link_template = None, None

    for joint in root.findall('fruit_joint_set'):
        name = joint.get('name')
        if name == "fruit_joint":
            fruit_joint_template = copy.deepcopy(joint)
            root.remove(joint)

    for link in root.findall('link'):
        name = link.get('name')
        if name == "default_fruit_link_name":
            fruit_link_template = copy.deepcopy(link)
            root.remove(link)

    return fruit_joint_template, fruit_link_template


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

    print(f"Successfully wrote the urdf to {path}.")
