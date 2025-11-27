from enum import Enum
from anytree import NodeMixin


class JointType(str, Enum):
    revolute = 'revolute'
    fixed = 'fixed'
    spherical = 'spherical'
    prismatic = 'prismatic'


class NodeType(str, Enum):
    branch = 'branch'
    leaf = 'leaf'
    fruit = 'fruit'

    @classmethod
    def from_node(cls, node):
        if isinstance(node, TreeBranch) or isinstance(node, Twig):
            return NodeType.branch
        elif isinstance(node, Leaf) or isinstance(node, LeafConn):
            return NodeType.leaf
        elif isinstance(node, Fruit) or isinstance(node, FruitConn):
            return NodeType.fruit
        else:
            raise ValueError(f"'{node}' is not a valid Node")



class Twig:
    pass


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

        self.material_rgba, self.material_name = self.get_joint_color(self.joint_type)

        # mass of all children including this branch/total mass of tree at first non-fixed joint
        self.branch_angle_rpy = branch_angle_rpy
        self.branch_rotation_axis = branch_rotation_axis

        self.idx = self.__class__.unique + 1
        self.level = self.parent.level + 1 if parent is not None else 0
        self.name = f'branch-B{self.idx}L{self.level}P{parent.idx}' \
            if parent is not None else f'trunk-B{self.idx}L{self.level}'
        self.child_idx = child_idx
        self.__class__.unique += 1

    @staticmethod
    def get_joint_color(_joint_type):
        # 'red'  # [0.201, 0.002, 0.071,1]
        if _joint_type == JointType.fixed:
            return [0.1, 0.2, 0.2, 1], 'brown'
        elif _joint_type == JointType.spherical:
            # return [0.8, 0.1, 0.8, 1], 'purple'
            return [0.1, 0.8, 0.2, 1], 'green'
        else:  # for JointType.revolute or any other type
            return [0.1, 0.8, 0.2, 1], 'green'


class Fruit:

    def __init__(self, fruit_origin_xyz, fruit_angle_rpy, child_idx):
        # true xyz/rpy of the fruit, doesn't include the spherical/prismatic orientations.
        self.fruit_origin_xyz = fruit_origin_xyz
        self.fruit_angle_rpy = fruit_angle_rpy
        self.radius = 0.02  # currently fixed
        self.child_idx = child_idx


class FruitConn(Fruit, NodeMixin):
    unique = -1

    def __init__(self, fruit_origin_xyz, fruit_angle_rpy, child_idx, parent=None):
        super(FruitConn, self).__init__(fruit_origin_xyz, fruit_angle_rpy, child_idx)

        self.idx = self.__class__.unique + 1
        self.level = parent.level + 1
        self.name = f'fruit-F{self.idx}T{self.level}P{parent.idx}'
        self.__class__.unique += 1


class Leaf:

    def __init__(self, leaf_angle_rpy, child_idx):
        self.leaf_angle_rpy = leaf_angle_rpy
        self.leaf_rotation_axis = (0, 1, 0)  # just random as leaf is a fixed joint
        self.child_idx = child_idx


class LeafConn(Leaf, NodeMixin):
    unique = -1

    def __init__(self, leaf_angle_rpy, child_idx, parent=None):
        super(LeafConn, self).__init__(leaf_angle_rpy, child_idx)

        self.idx = self.__class__.unique + 1
        self.level = parent.level + 1
        self.name = f'leaf-L{self.idx}T{self.level}P{parent.idx}'
        self.__class__.unique += 1
