from enum import Enum
import torch


class CollisionPenalty(Enum):
    NO_PENALTY = 0
    BINARY_PENALTY = 1
    DECAY_PENALTY = 2

    @classmethod
    def ptype(cls, val):
        for mem in CollisionPenalty:
            if mem.value == val:
                return mem
        raise ValueError(f"No CollisionPenalty member with value: {val}")


class SuccTracker:
    def __init__(self):
        # Initialize success/failure counts
        self.success = 0
        self.failure = 0

        # 1/0 indicating success or failure for each experiment
        self.succ_status = []
        # no of steps taken to reach succ/fail first for each experiment
        self.steps_to_succ_status = []

        # no of steps where the arm is in succ state for each experiment, e.g. for exposure
        self.steps_in_succ_status = []

        # the total/sum test rewards received for each experiment.
        self.test_rewards_status = []

    def _incr_success(self, count=1):
        # Increment success count
        self.success += count

    def _incr_failure(self, count=1):
        # Increment failure count
        self.failure += count

    def incr(self, succ_mask):
        """ pass a list like [1,0,1] indicating [succ,fail, succ] e.t.c"""
        assert all(elem in (0, 1) for elem in succ_mask), "List contains elements other than 0 or 1."
        self._incr_success(sum(succ_mask))
        self._incr_failure(len(succ_mask) - sum(succ_mask))
        self.succ_status.extend(succ_mask)

    def add_steps_to_succ(self, steps_to_succ):
        self.steps_to_succ_status.extend(steps_to_succ)

    def add_steps_in_succ(self, steps_in_succ):
        self.steps_in_succ_status.extend(steps_in_succ)

    def add_test_rewards(self, test_rewards):
        test_rewards = [round(_i, 2) for _i in test_rewards]
        self.test_rewards_status.extend(test_rewards)

    def get_counts(self):
        # Return a dictionary with success and failure counts
        if len(self.steps_to_succ_status) != 0:
            assert len(self.steps_to_succ_status) == len(self.succ_status), f"Invalid: {len(self.steps_to_succ_status)}"

        if len(self.steps_in_succ_status) != 0:
            assert len(self.steps_in_succ_status) == len(self.succ_status), f"Invalid: {len(self.steps_in_succ_status)}"

        assert all(elem in (0, 1) for elem in self.succ_status), "List contains elements other than 0 or 1."
        return {'success': self.success, 'failure': self.failure, 'steps_to_succ_status': self.steps_to_succ_status,
                'steps_in_succ_status': self.steps_in_succ_status, 'succ_status': self.succ_status,
                'test_rewards_status': self.test_rewards_status}

    def get_success_percentage(self):
        # Calculate percentage of successes
        total_attempts = self.success + self.failure
        if total_attempts == 0:
            raise ValueError("Invalid attempt")
        return (self.success / total_attempts) * 100.0


class MultiSuccTracker:
    """ To track success/failure status when tests are conducted for multiple envs in parallel"""

    def __init__(self, num_test_envs, num_test_targets, context_folder=None, num_obs=-1,
                 max_episode_length=-1, brush_past_norm_cf=-1):
        self.num_test_envs = num_test_envs
        self.num_test_targets = num_test_targets

        # training/test information for future comparison
        self.context_folder = context_folder
        self.num_obs = num_obs
        self.max_episode_length = max_episode_length
        self.brush_past_norm_cf = brush_past_norm_cf

        self._mSuccTracker = []
        # for each test target create a success tracker
        # list of succ_trackers of size num_targets (voxels / e.g. 60)
        # Each tracker has succ_status/steps_to_succ_status//steps_in_succ_status = list of size num_envs

        # a place holder for additional metrics to be used for different tasks. Only final values are scored
        # the score computation is done in RL itself.
        self.task_specific_metrics = {}

    def incr(self, succ_mask):
        # this should be called only once for a given test target.
        self._mSuccTracker.append(SuccTracker())
        assert len(self._mSuccTracker) <= self.num_test_targets

        _msg = f"For each test trajectory, no of successes should num_envs : {len(succ_mask)}"
        assert len(succ_mask) == self.num_test_envs, _msg
        self._mSuccTracker[-1].incr(succ_mask)
        assert len(self._mSuccTracker[-1].succ_status) == self.num_test_envs, _msg

    def add_steps_to_succ(self, steps_to_succ):
        _msg = f"For each test trajectory, no of successes should num_envs : {len(steps_to_succ)}"
        assert len(steps_to_succ) == self.num_test_envs, _msg
        self._mSuccTracker[-1].add_steps_to_succ(steps_to_succ)
        assert len(self._mSuccTracker[-1].steps_to_succ_status) == self.num_test_envs, _msg

    def add_steps_in_succ(self, steps_in_succ):
        _msg = f"For each test trajectory, no of successes should num_envs : {len(steps_in_succ)}"
        assert len(steps_in_succ) == self.num_test_envs, _msg
        self._mSuccTracker[-1].add_steps_in_succ(steps_in_succ)
        assert len(self._mSuccTracker[-1].steps_in_succ_status) == self.num_test_envs, _msg

    def add_test_rewards(self, test_rewards):
        _msg = f"For each test trajectory, no of test rewards should == num_envs : {len(test_rewards)}"
        assert len(test_rewards) == self.num_test_envs, _msg
        self._mSuccTracker[-1].add_test_rewards(test_rewards)
        assert len(self._mSuccTracker[-1].test_rewards_status) == self.num_test_envs, _msg

    def _get_counts(self, target_idx):
        # Note: below target_idx values will differ with curr_target_idx by 1 as curr_target_idx is 1-indexed
        return self._mSuccTracker[target_idx].get_counts()

    def _get_success_percentage(self, target_idx):
        return self._mSuccTracker[target_idx].get_success_percentage()

    def get_overall_success_percentage(self):
        success_percentages = []
        for target_idx in range(self.num_test_targets):
            try:
                success_percentages.append(
                    self._mSuccTracker[target_idx].get_success_percentage()
                )
            except ValueError:
                continue

        if not success_percentages:
            raise ValueError("No valid attempts across environments")

        return sum(success_percentages) / len(success_percentages)

    def last_test_succ_metrics(self):
        return {
            'succ_pct': self._mSuccTracker[-1].get_success_percentage(),
            'succ_status': self._mSuccTracker[-1].succ_status,
            'steps_to_succ_status': self._mSuccTracker[-1].steps_to_succ_status,
            'steps_in_succ_status': self._mSuccTracker[-1].steps_in_succ_status,
            'test_rewards_status': self._mSuccTracker[-1].test_rewards_status
        }

    def get_overall_counts(self, include_each_target_counts=True):
        overall_success = 0
        overall_failure = 0
        overall_steps_to_succ_status = []
        overall_steps_in_succ_status = []
        overall_test_rewards_status = []
        overall_succ_status = []

        for target_idx in range(self.num_test_targets):
            counts = self._mSuccTracker[target_idx].get_counts()
            overall_success += counts['success']
            overall_failure += counts['failure']
            overall_steps_to_succ_status.append(counts['steps_to_succ_status'])
            overall_steps_in_succ_status.append(counts['steps_in_succ_status'])
            overall_test_rewards_status.append(counts['test_rewards_status'])
            overall_succ_status.append(counts['succ_status'])

            if len(counts['steps_to_succ_status']) != 0:
                assert len(counts['steps_to_succ_status']) == len(counts['succ_status']), f"Invalid len"

            if len(counts['steps_in_succ_status']) != 0:
                assert len(counts['steps_in_succ_status']) == len(counts['succ_status']), f"Invalid len"

        final_results = {
            'success': overall_success,
            'failure': overall_failure,
            'succ_pct': round((overall_success / (overall_failure + overall_success)) * 100.0, 2),
            'steps_to_succ_status': overall_steps_to_succ_status,
            'steps_in_succ_status': overall_steps_in_succ_status,
            'test_rewards_status': overall_test_rewards_status,
            'succ_status': overall_succ_status,
            'num_obs': self.num_obs,
            'context_folder': self.context_folder,
            'max_episode_length': self.max_episode_length,
            'brush_past_norm_cf': self.brush_past_norm_cf,
            'mean_test_rewards_status': torch.tensor(overall_test_rewards_status).mean(),
            'mean_steps_in_succ_status': torch.tensor(overall_steps_in_succ_status).float().mean()
        }

        if self.task_specific_metrics is not None and len(self.task_specific_metrics) != 0:
            final_results.update(self.task_specific_metrics)

        if include_each_target_counts:
            for target_idx in range(self.num_test_targets):
                final_results[f'target{target_idx}'] = self._mSuccTracker[target_idx].get_counts()

        return final_results


class NetGenForceTracker:
    """ To track net generalised forces (contact forces, dof torques e.t.c) ,
        when tests are conducted for multiple envs in parallel"""

    def __init__(self, _num_test_envs, num_test_targets, max_episode_length, num_envs):
        self.num_test_envs = _num_test_envs
        self.num_test_targets = num_test_targets
        self.max_episode_length = max_episode_length
        self.num_envs = num_envs

        # list of tensors; num_targets (voxels / e.g. 60) x num_envs x episode_len(traj_len / e.g.800)
        # Each tensor is num_envs x episode_len(traj_len / e.g.800)
        self._mNetGfTracker = []

    def create_empty_test_traj(self, device):
        if len(self._mNetGfTracker) > 1:
            self.clean_up()
            exp_shape = torch.Size([self.num_envs, self.max_episode_length])
            assert self._mNetGfTracker[-1].shape == exp_shape, f"Invalid Shape: {self._mNetGfTracker[-1].shape}"

        # add empty tensor for the upcoming trajectory.
        self._mNetGfTracker.append(torch.empty(0, device=device))

    def append_test_trajs(self, net_gf_mag):
        assert len(net_gf_mag) == self.num_envs, f"Invalid shape, expecting torch.Size([num_envs]): {len(net_gf_mag)}"
        net_gf_mag = net_gf_mag.unsqueeze(1)  # Shape [1]
        self._mNetGfTracker[-1] = torch.cat((self._mNetGfTracker[-1], net_gf_mag), dim=1)
        updated_traj_len = self._mNetGfTracker[-1].shape[1]
        assert updated_traj_len <= self.max_episode_length + 1, f"Invalid len: {updated_traj_len}"

    def last_test_gf_metrics(self):
        if len(self._mNetGfTracker[-1]) < 1:
            raise ValueError("No tests are performed yet")

        # if this cleanup is not called the final test may have 1  step extra.
        self.clean_up()
        exp_shape = torch.Size([self.num_envs, self.max_episode_length])
        assert self._mNetGfTracker[-1].shape == exp_shape, f"{self._mNetGfTracker[-1].shape}"

        return {
            'average': torch.mean(self._mNetGfTracker[-1]),
            'sum': torch.sum(self._mNetGfTracker[-1]),
        }

    def clean_up(self):
        # sometimes the traj length becomes 801 or 501, with one step extra. Chuck it.
        # should not clean up if the length is < 799 or  > 801
        if self._mNetGfTracker[-1].shape == torch.Size([self.num_envs, self.max_episode_length + 1]):
            self._mNetGfTracker[-1] = self._mNetGfTracker[-1][:, :self.max_episode_length]

    def net_gf_metric_t(self, device):
        assert len(self._mNetGfTracker) >= self.num_test_targets, "First execute  all test trajectories"
        gf_metrics_t = torch.stack(self._mNetGfTracker).to(device)
        exp_shape = torch.Size([self.num_test_targets, self.num_envs, self.max_episode_length])
        assert gf_metrics_t.shape == exp_shape, f"Invalid Shape: {gf_metrics_t.shape}"
        return gf_metrics_t
