import pandas as pd
import bisect


def zip_frames_by_close_match(pose_ts_pairs, wrench_ts_pairs):
    def _append_row(df, _row):
        return pd.concat([df,
                          pd.DataFrame([_row], columns=_row.index)]
                         ).reset_index(drop=True)

    pose_ts_keys = list(pose_ts_pairs.keys())
    combined_tool_frame = pd.DataFrame(columns=('force.time', 'pose.time', 'wrench', 'pose'))

    for force_ts, force_z in wrench_ts_pairs.items():
        # give the closest key that is greater than the search key, not the closest key, guess that is OK
        closest_ts_key_idx = bisect.bisect(pose_ts_keys, force_ts)

        # occasionally first/last row would be missing in one of the dfs.
        if closest_ts_key_idx == len(pose_ts_keys) or closest_ts_key_idx == 0:
            continue

        # this is for the staggered force/pos by 1/10th sec issue. simply sh*t
        if closest_ts_key_idx != 0:
            closest_pose_ts_key1 = pose_ts_keys[closest_ts_key_idx]
            closest_pose_ts_key2 = pose_ts_keys[closest_ts_key_idx - 1]
            if abs(closest_pose_ts_key1 - force_ts) < abs(closest_pose_ts_key2 - force_ts):
                closest_pose_ts_key = closest_pose_ts_key1
            else:
                closest_pose_ts_key = closest_pose_ts_key2

        assert closest_pose_ts_key is not None
        row = pd.Series({'force.time': force_ts,
                         'pose.time': closest_pose_ts_key,
                         'wrench': force_z,
                         'pose': pose_ts_pairs[closest_pose_ts_key],
                         })
        combined_tool_frame = _append_row(combined_tool_frame, row)

    time_diff = combined_tool_frame['force.time'] - combined_tool_frame['pose.time']
    assert (time_diff.abs() > 0.2).any() == False, "Large time difference b/w pose and force timestamps"

    return combined_tool_frame


def conform_shapes(combined_tool_frames):
    """ Make all the individual dataframes of the same size"""
    original_traj_lens = [frame.shape[0] for _, frame in combined_tool_frames.items()]
    lowest_len = min(original_traj_lens)
    print(f"\nWARN: Updating traj lengths from {original_traj_lens} to {lowest_len}")
    return {exp_idx: frame[:lowest_len] for exp_idx, frame in combined_tool_frames.items()}
