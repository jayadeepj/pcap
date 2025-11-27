#! /usr/bin/env python
"""Script to perform sim-to-real transactions for the RL tactile task.

1. Read Kinova DOF topics for pos, vel, torques
2. Reads Kinova EE pose
3. Exposes the read kinova metrics via a GET REST Service

4. Stands up a POST REST Service that will accept dof pos (joint angles)
5. Sets the joint angles on the real arm via ros topics

Run Instructions

# Use 1 to capture trajectory
Note 1: rosrun spot_control <file_name>.py j2n6s300 0

"""

import rospy
import roslib;

from utils import path_setup
import tactile_rl_rest_srvc as rest
from utils.kinova_helper import *
from estimation.helpers import misc
import argparse

roslib.load_manifest('spot_control')
misc.set_seed()


def argument_parser(argument_):
    """ Argument parser """
    parser = argparse.ArgumentParser(description='Drive robot end-effector to command Cartesian pose')
    parser.add_argument('kinova_robot_type', metavar='kinova_robot_type', type=str, default='j2n6a300',
                        help='kinova_robot_type is in format of: [{j|m|r|c}{1|2}{s|n}{4|6|7}{s|a}{2|3}{0}{0}]. '
                             'eg: j2n6a300 refers to jaco v2 6DOF assistive 3fingers. '
                             'Please  that not all options are valid for different robot types.')

    parser.add_argument('capture_cmd_traj', metavar='capture_cmd_traj', type=int, default=0,
                        help='capture_cmd_traj is in boolean format to save commanded vel/pos trajectory')

    parser.add_argument('capture_exec_traj', metavar='capture_exec_traj', type=int, default=0,
                        help='capture_exec_traj is in boolean format to save executed vel/pos/torque trajectory')

    args = parser.parse_args(argument_)
    return args


def runner(_args_):
    arm = kinova_robot_type_parser(_args_.kinova_robot_type)
    rospy.init_node(arm.prefix + 'tactile_rl_txn_control')
    rest.stand_up_service(port=3738, _arm=arm, capture_cmd_traj=bool(_args_.capture_cmd_traj),
                          capture_exec_traj=bool(_args_.capture_exec_traj))


# use  store_rand_mdeg_poses(count=?) to store the random poses, will overwrite the current file.
if __name__ == '__main__':
    args_ = argument_parser(None)
    runner(args_)
