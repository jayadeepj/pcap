class LShape:
    """ Link cylinder shapes"""

    def __init__(self, length, radius):
        self.length = length
        self.radius = radius

    def __str__(self):
        return f" length {self.length}, radius {self.radius}"


class TreeMetaAttrs:
    """ To share the tree meta attributes shapes across projects"""

    def __init__(self, urdf_path, l_shape_dict, dof_root):
        self.urdf_path = urdf_path
        self.l_shape_dict = l_shape_dict
        self.dof_root = dof_root
