from bagpy import bagreader
import pandas as pd
import collections
from estimation.real.domain import Pose, Wrench


def read_bags(bag_path):
    b = bagreader(bag_path)
    for t in b.topics:
        if 'tool_pose' in t:
            tool_pose_df = pd.read_csv(b.message_by_topic(t))
        elif 'tool_wrench' in t:
            tool_wrench_df = pd.read_csv(b.message_by_topic(t))
        else:
            raise ValueError("Unhandled topic in ROS Bag")

    assert tool_pose_df is not None and tool_wrench_df is not None

    tool_pose_df.sort_values(by=['Time'], inplace=True)
    tool_wrench_df.sort_values(by=['Time'], inplace=True)

    tool_pose_df.columns = tool_pose_df.columns.str.replace(r'.', '_', regex=True)
    tool_wrench_df.columns = tool_wrench_df.columns.str.replace(r'.', '_', regex=True)

    wrenches = [Wrench(fx=0, fy=0, fz=tw.wrench_force_z,
                       tx=tw.wrench_torque_x, ty=tw.wrench_torque_y, tz=tw.wrench_torque_z)
                for tw in tool_wrench_df.itertuples()]

    wrench_ts_pairs = collections.OrderedDict(zip(tool_wrench_df['Time'], wrenches))

    poses = [Pose(x=0, y=0, z=tp.pose_position_z,
                  ox=tp.pose_orientation_x, oy=tp.pose_orientation_y, oz=tp.pose_orientation_z)
             for tp in tool_pose_df.itertuples()]

    pose_ts_pairs = collections.OrderedDict(zip(tool_pose_df['Time'], poses))

    return pose_ts_pairs, wrench_ts_pairs
