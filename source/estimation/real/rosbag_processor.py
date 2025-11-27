""" Converts ROS Bags to Ground Truth experiments

Run Instructions

Note 1:  Set arm_type to kinova/franka below before runs
Note 2: Modify the file paths and names if required
Note 3 : Invoke this file from command line as python source/estimation/real/rosbag_processor.py

"""

import pickle
import os

import torch
import math
from utils import path_setup
from estimation.math import common

import tool_frame
from estimation.real.domain import MeasurementConstants
from estimation.real.domain import ArmType
from estimation.isaac.profile import Experiment
from estimation.real.franka import franka_bag_handler
from estimation.real.kinova import kinova_bag_handler

arm_type = ArmType.kinova
bags_path_root = f'data/rosbags/external/{arm_type.value}/tree/set_p4'
ser_path_root = f'data/serialized/external/{arm_type.value}/tree/bags/set_p4'
ser_file_prefix = f'8a_tree_artifact_pos_ctrl_'
force_conform_shapes = True  # set True if all trajectories need to be truncated to a common length
mc = MeasurementConstants(arm_type=arm_type)


# Note: Invoke this file from command line as python source/estimation/real/rosbag_processor.py

def read_bag_files():
    arm_bag_path = os.path.join(path_setup.notebooks_dir, f'{bags_path_root}/')
    bag_set = []
    for p in os.listdir(arm_bag_path):
        file_path = os.path.join(arm_bag_path, p)
        if '.bag' in str(p):
            bag_set.append(file_path)
    return sorted(bag_set)


def velocity_from_position(time_trajectory, pos_trajectory):
    """ Compute linear velocity of the arm from position, time ignoring the orientation"""
    return torch.gradient(pos_trajectory, spacing=(time_trajectory,))[0]


def gen_trajectories(frame):
    pos_trajectory, time_trajectory, force_trajectory = torch.empty(0), torch.empty(0), torch.empty(0)

    link_world_pos_start = frame.loc[0, 'pose'].pose()  # get the initial point, this can vary b/w grasps
    for _, row in frame.iterrows():
        force_state = row['wrench'].force()
        link_world_pos_end = row['pose'].pose()
        _current_pose = link_world_pos_end

        curr_pos_deformations = common.l2_norm(link_world_pos_end[:, :3], link_world_pos_start[:, :3])
        pos_trajectory = torch.cat((pos_trajectory, curr_pos_deformations), dim=0)

        _ctime = row['force.time']
        current_time = torch.DoubleTensor([_ctime])
        time_trajectory = torch.cat((time_trajectory, current_time), dim=0)

        force_trajectory = torch.cat((force_trajectory, force_state), dim=0)

    vel_trajectory = velocity_from_position(time_trajectory, pos_trajectory)

    pos_trajectory = pos_trajectory.unsqueeze(1)
    time_trajectory = time_trajectory.unsqueeze(1)
    vel_trajectory = vel_trajectory.unsqueeze(1)

    print(f"pos_trajectory.shape: {pos_trajectory.shape}")
    print(f"vel_trajectory.shape: {vel_trajectory.shape}")
    print(f"force_trajectory.shape: {force_trajectory.shape}")
    print(f"time_trajectory.shape: {time_trajectory.shape}")

    trajectory_length = pos_trajectory.shape[0]
    assert pos_trajectory.shape[0] == force_trajectory.shape[0], "Invalid Trajectory Shape"
    assert pos_trajectory.shape == torch.Size([trajectory_length, 1]), "Invalid Trajectory Shape"
    assert pos_trajectory.shape[0] == vel_trajectory.shape[0], "Invalid Trajectory Shape"

    # check if real time taken in sec and the expected freq are close to 3 sec tolerance
    total_real_time_taken_sec = (time_trajectory[-1] - time_trajectory[0]).item()
    assert math.isclose(trajectory_length * mc.arm_dt, total_real_time_taken_sec,
                        abs_tol=5e0), f"Freq Mismatch: {total_real_time_taken_sec}"

    return force_trajectory, pos_trajectory, vel_trajectory, time_trajectory


def generator():
    # read all rosbags
    bag_path_set = read_bag_files()

    def _index_frames(bag_path):
        _exp_idx = bag_path.split(r'/')[-1].replace('.bag', '')
        if arm_type == ArmType.kinova:
            pose_ts_pairs, wrench_ts_pairs = kinova_bag_handler.read_bags(bag_path)
        elif arm_type == ArmType.franka:
            pose_ts_pairs, wrench_ts_pairs = franka_bag_handler.read_bags(bag_path)

        _combined_tool_frame = tool_frame.zip_frames_by_close_match(pose_ts_pairs, wrench_ts_pairs)
        return _exp_idx, _combined_tool_frame

    _all_frames = [_index_frames(bag_path) for bag_path in bag_path_set]
    all_frames = {k: v for (k, v) in _all_frames}

    cleaned_frames = tool_frame.conform_shapes(all_frames) if force_conform_shapes else all_frames

    for exp_idx, frame in cleaned_frames.items():
        print(f"Process File for exp_idx: {exp_idx}")

        force_trajectory, pos_trajectory, vel_trajectory, time_trajectory = gen_trajectories(frame)
        exp = Experiment(link_name='physical',
                         force_profile=None,
                         true_pos_trajectory=pos_trajectory,
                         true_vel_trajectory=vel_trajectory,
                         true_force_trajectory=force_trajectory,
                         noise=None,
                         time_trajectory=time_trajectory,
                         mc=mc.__dict__)

        gt_artifact_ser = {
            'b1_experiment': [exp]
        }

        with open(os.path.join(path_setup.notebooks_dir,
                               f'{ser_path_root}/{ser_file_prefix}{exp_idx}.pickle'),
                  'wb') as handle:
            pickle.dump(gt_artifact_ser, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    generator()
