import collections
from bagpy import bagreader
import pandas as pd
import collections
from estimation.real.domain import Pose, Wrench


def read_bags(bag_path):
    b = bagreader(bag_path)
    for t in b.topics:
        if 'franka_states' in t:
            tool_pose_df = pd.read_csv(b.message_by_topic(t))
        else:
            continue  # there may be other topics that we are not interested in.

    assert tool_pose_df is not None

    tool_pose_df = tool_pose_df[tool_pose_df.index % 10 == 0]
    tool_pose_df.sort_values(by=['Time'], inplace=True)
    tool_pose_df.columns = tool_pose_df.columns.str.replace(r'.', '_', regex=True)

    # Note: we are setting forces in all other axis to be zero.
    wrenches = [Wrench(fx=0, fy=0, fz=tp.O_F_ext_hat_K_2, tx=0, ty=0, tz=0)
                for tp in tool_pose_df.itertuples()]

    wrench_ts_pairs = collections.OrderedDict(zip(tool_pose_df['Time'], wrenches))

    poses = [Pose(x=0, y=0, z=tp.O_T_EE_14, ox=0, oy=0, oz=0)
             for tp in tool_pose_df.itertuples()]

    pose_ts_pairs = collections.OrderedDict(zip(tool_pose_df['Time'], poses))
    return pose_ts_pairs, wrench_ts_pairs
