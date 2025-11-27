from typing import List
from estimation.math.uncertainty.prior import SmoothedBoxPrior
from estimation.isaac.param import JointParams
from estimation.nn.prior import NNInequalityPrior
from estimation.isaac.mechanics import DeformationKwargs
import copy
import torch


class ParamDensity:
    def __init__(self, sim_tree_handle, kt, mechanics, all_stable_states, def_kwargs: List[DeformationKwargs],
                 target_dof, target_p_damping_temp, other_jp: List[JointParams], sb_prior: SmoothedBoxPrior,
                 ineq_prior: NNInequalityPrior = None):

        self.sim_tree_handle = sim_tree_handle
        self.kt = kt  # temperature x B hyperparameter for Boltzmann probability measure

        self.mechanics = mechanics
        self.def_kwargs = def_kwargs  # the attributes required for simulation & deformation

        self.target_dof = target_dof  # the dof for which the x tensor parameters are for
        self.target_p_damping_temp = target_p_damping_temp  # this is just temp in case we are estimating only stiffness
        self.other_jp = other_jp  # all other joint parameters except the current; in case of single branch estimation.

        self.sb_prior = sb_prior  # Smooth box prior
        self.ineq_prior = ineq_prior  # Inequality prior, can be left none

        self.all_stable_states = all_stable_states

    def _cost_fx(self, x):
        """ f: R ^ d => R; Parameter tensor In , Cost tensor Out. In shape (n X param_count)== Out.shape (n) """

        # This negative problem might be overcome by the smooth box gradient
        assert torch.all(torch.gt(x, -5.0)).item() is True, f"Parameters must be non-negative{x}"

        assert x.shape[1] % 2 == 0 or x.shape[1] == 1, " Currently Implemented only for 2 params/per branch"
        if x.shape[1] > 1:
            assert x.shape[1] / 2 == len(self.target_dof), "Invalid number of target dofs"

        if x.shape[1] not in [1, 2, 4, 6]:
            assert self.ineq_prior is None, "Currently inequality prior is only for <2 or 3> branches x 2 param combo"

        # flatten each dimension
        if x.shape[1] == 1:
            x_0 = x[:, 0]
            x_1 = self.target_p_damping_temp[:, 0]
            target_jp_0 = JointParams.from_tensor(self.target_dof[0], x_0, x_1)
            if self.other_jp is None or len(self.other_jp) == 0:
                # to estimate only 1 branch 1x parameters
                joint_params = [target_jp_0]
            else:
                #  to estimate 1x b2 parameters by fixing 1x b1 parameters
                joint_params = sorted([target_jp_0, *self.other_jp], key=lambda jp: jp.dof_id)
        elif x.shape[1] == 2:
            x_0, x_1 = x[:, 0], x[:, 1]
            target_jp_0 = JointParams.from_tensor(self.target_dof[0], x_0, x_1)
            if self.other_jp is None or len(self.other_jp) == 0:
                # to estimate only 1 branch 2x parameters
                joint_params = [target_jp_0]
            else:
                # to estimate 2x b2 parameters by fixing 2x b1 parameters
                joint_params = sorted([target_jp_0, *self.other_jp], key=lambda jp: jp.dof_id)
        elif x.shape[1] == 4:
            x_0, x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            target_jp_0 = JointParams.from_tensor(self.target_dof[0], x_0, x_1)
            target_jp_1 = JointParams.from_tensor(self.target_dof[1], x_2, x_3)
            joint_params = sorted([target_jp_0, target_jp_1], key=lambda jp: jp.dof_id)
        elif x.shape[1] == 6:
            x_0, x_1, x_2, x_3, x_4, x_5 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4], x[:, 5]
            target_jp_0 = JointParams.from_tensor(self.target_dof[0], x_0, x_1)
            target_jp_1 = JointParams.from_tensor(self.target_dof[1], x_2, x_3)
            target_jp_2 = JointParams.from_tensor(self.target_dof[2], x_4, x_5)
            joint_params = sorted([target_jp_0, target_jp_1, target_jp_2], key=lambda jp: jp.dof_id)
        else:
            raise ValueError("Unimplemented for branch sequences > 3")

        fx = self.mechanics \
            .sim_multi_deformations(sim_tree_handle=self.sim_tree_handle,
                                    joint_params=joint_params,
                                    all_stable_states=self.all_stable_states,
                                    kwargs=self.def_kwargs)

        assert x.shape[0] == fx.shape[0]
        assert fx.shape[1] == 1
        return x_0, x_1, fx

    def _log_likelihood_px(self, x):
        """ Log of unnormalized probability density p(D|x)"""
        x_0, x_1, fx = self._cost_fx(x)
        px = -1 * fx / self.kt
        return x_0, x_1, fx, px

    def log_clipped_px(self, x):
        """ Return clipped density: log {smooth box prob * p(D|x) }
         Warning: The values for the range
         """

        self.mechanics.logging.debug(f"x: {x}")
        x_0, x_1, fx, px = self._log_likelihood_px(x)

        # Not sure if this copy is required, but it seems like there is some in-state modification in sb. Wouldn't hurt.
        sm = copy.deepcopy(self.sb_prior)
        sm_box_prob = sm.log_prob(x.unsqueeze(1))

        if self.ineq_prior and x.shape[1] == 4:
            self.mechanics.logging.info(f"Using Inequality Priors to compute p(x) * p0(x)")
            ineq_prior_st = self.ineq_prior.evaluate(x[:, (0, 2)])
            ineq_prior_da = self.ineq_prior.evaluate(x[:, (1, 3)])

            self.mechanics.logging.debug(f"Stiffness Inequality Prior {ineq_prior_st}")
            self.mechanics.logging.debug(f"Damping Inequality Prior {ineq_prior_da}")
            cpx = px + ineq_prior_st + ineq_prior_da + sm_box_prob
        elif self.ineq_prior and x.shape[1] == 6:
            ineq_prior_st1 = self.ineq_prior.evaluate(x[:, (0, 2)])
            ineq_prior_da1 = self.ineq_prior.evaluate(x[:, (1, 3)])
            ineq_prior_st2 = self.ineq_prior.evaluate(x[:, (2, 4)])
            ineq_prior_da2 = self.ineq_prior.evaluate(x[:, (3, 5)])

            self.mechanics.logging.debug(f"Stiffness Inequality Prior1 {ineq_prior_st1}")
            self.mechanics.logging.debug(f"Damping Inequality Prior1 {ineq_prior_da1}")
            self.mechanics.logging.debug(f"Stiffness Inequality Prior2 {ineq_prior_st2}")
            self.mechanics.logging.debug(f"Damping Inequality Prior2 {ineq_prior_da2}")
            cpx = px + ineq_prior_st1 + ineq_prior_da1 + ineq_prior_st2 + ineq_prior_da2 + sm_box_prob

        else:
            self.mechanics.logging.info(f"Skipping Inequality Priors, only add the smooth box prior")
            cpx = px + sm_box_prob

        return x_0, x_1, fx, px, cpx

    def log_solo_clipped_px(self, x):
        x_0, x_1, fx, px, cpx = self.log_clipped_px(x)
        return cpx

    def neg_log_solo_clipped_px(self, x):
        return - 1 * self.log_solo_clipped_px(x)
