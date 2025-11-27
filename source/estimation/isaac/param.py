import numpy as np
from enum import Enum
import itertools
import torch


class ParamClass(str, Enum):
    stiffness = 'stiffness'
    damping = 'damping'


class ParamType:
    def __init__(self, pclass: ParamClass, default_val):
        # stiffness/damping
        self.pclass = pclass
        # default value of the parameter for branches not being deformed.
        self.default_val = default_val


class StiffnessType(ParamType):

    def __init__(self):
        ParamType.__init__(self, pclass=ParamClass.stiffness, default_val=10000.)


class DampingType(ParamType):

    def __init__(self):
        ParamType.__init__(self, pclass=ParamClass.damping, default_val=100.)


class Param:
    """ ParamType:Param :: 1:M  """

    def __init__(self, param_vals: np.ndarray, param_type: ParamType):
        self.param_vals = param_vals
        self.param_type = param_type

    some_max_param_value = 1e7

    @classmethod
    def from_range(cls, count, lower, upper, param_type: ParamType):
        return cls(param_vals=cls.gen_vals_from_range(count, lower, upper), param_type=param_type)

    @classmethod
    def from_fixed_val(cls, count, value, param_type: ParamType):
        return cls(param_vals=cls.gen_vals_from_fixed_val(count, value), param_type=param_type)

    @staticmethod
    def gen_vals_from_range(count, lower, upper):
        return np.linspace(start=lower, stop=upper, num=count)

    @staticmethod
    def gen_vals_from_fixed_val(count, value):
        return np.full(count, value)

    def param_len(self):
        return len(self.param_vals)

    def __repr__(self):
        return f" Param (param_type:{self.param_type.pclass}, param_vals:{str(self.param_vals)})"

    def next_slice(self, curr_idx, slice_length):
        return [self.param_vals[curr_idx + i]
                if curr_idx + i < len(self.param_vals) else Param.some_max_param_value
                for i in range(slice_length)]


class JointParams:
    """ Combination of DOFs and Joint Parameters"""

    def __init__(self, dof_id: int, jp_stiffness: Param, jp_damping: Param):
        self.dof_id = dof_id

        self.jp_stiffness = jp_stiffness
        self.jp_damping = jp_damping

        if jp_stiffness.param_type.pclass != StiffnessType().pclass:
            raise ValueError("Invalid Parameter Type Class")

        if jp_damping.param_type.pclass != DampingType().pclass:
            raise ValueError("Invalid Parameter Type Class")

        if self.jp_stiffness.param_len() != self.jp_damping.param_len():
            raise ValueError("Length of parameter arrays of all param classes must match")

    def _x(self, device):
        # build joint parameter tensor
        kp = torch.from_numpy(self.jp_stiffness.param_vals).to(device).unsqueeze(1)
        kd = torch.from_numpy(self.jp_damping.param_vals).to(device).unsqueeze(1)
        return torch.concat((kp, kd), dim=1)

    @classmethod
    def from_cross_join_vals(cls, dof_id, p_stiffness_vals, p_damping_vals):

        # Find combinations of values
        st_mat, da_mat = np.meshgrid(
            np.sort(np.unique(p_stiffness_vals)),
            np.sort(np.unique(p_damping_vals)))

        jp_stiffness = Param(param_vals=st_mat.ravel(), param_type=StiffnessType())
        jp_damping = Param(param_vals=da_mat.ravel(), param_type=DampingType())

        return cls(dof_id=dof_id, jp_stiffness=jp_stiffness, jp_damping=jp_damping, )

    @classmethod
    def from_tensor(cls, dof_id, x_0, x_1):

        jp_stiffness = Param(param_vals=x_0.cpu().numpy(), param_type=StiffnessType())
        jp_damping = Param(param_vals=x_1.cpu().numpy(), param_type=DampingType())
        return cls(dof_id=dof_id, jp_stiffness=jp_stiffness, jp_damping=jp_damping)

    def param_len(self):
        return self.jp_stiffness.param_len()

    def __repr__(self):
        return f"JointParams(dof:{self.dof_id}, p_stiffness:{str(self.jp_stiffness)}, p_damping:{str(self.jp_damping)})"

    @staticmethod
    def random(kp_r1, kp_r2, kd_r1, kd_r2, size, branch_count, device):
        """ Generate random joint parameters for {branch_count} branches within specified limits.
        For each branch use stiffness & damping """

        def _single():
            st = ((kp_r1 - kp_r2) * (torch.rand(size, *torch.Size([1]))) + kp_r2).to(device=device)
            da = ((kd_r1 - kd_r2) * (torch.rand(size, *torch.Size([1]))) + kd_r2).to(device=device)
            return [st, da]

        jp = [_single() for _ in range(branch_count)]
        return torch.cat(tuple(itertools.chain(*jp)), dim=1)
