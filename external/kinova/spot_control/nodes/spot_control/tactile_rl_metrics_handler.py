""" Get individual Kinova ROS metrics from topics in parallel"""

import os
import time
import math
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from reinforcement.ige.real.kinova.robot_domain import KinovaMetricsR
import joint_state_handler
from datetime import datetime

spot_path = os.getenv('spot_path')
notebooks_dir = f"{spot_path}/notebooks/work2/"
_set = 'set_p4'


def all_km_metrics(arm, joint_state_tracker):
    ee_pose_tracker = EEPoseTracker(arm)  # only one instance will be created
    ee_pose = ee_pose_tracker.current_ee_pose()
    curr_dof_pos, curr_dof_vel, curr_dof_torque = joint_state_tracker.current_joint_state()

    return KinovaMetricsR(robot_dof_pos=curr_dof_pos,
                          robot_dof_vel=curr_dof_vel,
                          robot_dof_torque=curr_dof_torque,
                          hand_pos=ee_pose[0:3],
                          hand_rot=ee_pose[3:])


class EEPoseTracker:
    """ A Singleton class that fetches EE Pose"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, arm_):
        if not hasattr(self, 'initialized'):
            self.arm = arm_
            self._current_pose = [0] * 7  # 3 pos + 4 rotation
            self.req_submitted = False
            self.initialized = True

    @staticmethod
    def _parse_msg(geom_msg):
        # Extract position from PoseStamped message
        position = geom_msg.pose.position
        x = position.x
        y = position.y
        z = position.z

        # Extract orientation from PoseStamped message
        orientation = geom_msg.pose.orientation
        ox = orientation.x
        oy = orientation.y
        oz = orientation.z
        ow = orientation.w

        return [x, y, z, ox, oy, oz, ow]

    def _get_ee_pose(self):
        # This method (to submit ros node req) should be called only once
        print("Inside _get_ee_pose: req submitted", self.req_submitted)
        if not self.req_submitted:
            # wait to get current position
            topic_address = '/' + self.arm.prefix + 'driver/out/tool_pose'
            rospy.Subscriber(topic_address, PoseStamped, self._set_ee_pose)
            # the wait is blocking.
            rospy.wait_for_message(topic_address, PoseStamped, timeout=10)
            self.req_submitted = True

    def _set_ee_pose(self, geom_msg):
        # _set_current_pose gets called each time the topic has a message.
        # So it will be running in the background once it gets called once.
        self._current_pose = self._parse_msg(geom_msg)

    def current_ee_pose(self):
        self._get_ee_pose()
        return self._current_pose


class JointStateTracker:
    """ A Singleton class that fetches Joint State"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.dof_js_tracker = []
        return cls._instance

    def __init__(self, arm_, capture_exec_traj):
        if not hasattr(self, 'initialized'):
            self.arm = arm_
            self._current_dof_pos = [0] * self.arm.arm_joint_number
            self._current_dof_vel = [0] * self.arm.arm_joint_number
            self._current_dof_torque = [0] * self.arm.arm_joint_number
            self.req_submitted = False
            self.initialized = True
            self.capture_exec_traj = capture_exec_traj
            if capture_exec_traj:
                collision_path_root = f'data/rosbags/external/kinova/rl/collisions/{_set}/exec/js'
                self.collision_js_path = os.path.join(notebooks_dir, collision_path_root)
                print(f"Capturing Exec States at : {self.collision_js_path}.......")

    def _parse_msg(self, sensor_msg):
        _current_dof_pos_ = sensor_msg.position[0:self.arm.arm_joint_number]
        _current_dof_vel_ = sensor_msg.velocity[0:self.arm.arm_joint_number]
        _current_dof_torque_ = sensor_msg.effort[0:self.arm.arm_joint_number]

        return _current_dof_pos_, _current_dof_vel_, _current_dof_torque_

    def _get_joint_state(self):
        """/out/joint_state will give joint (rad), position, velocity (rad/s) and effort (Nm) information."""
        # This method (to submit ros node req) should be called only once
        print("Inside _get_joint_state: req submitted", self.req_submitted)
        if not self.req_submitted:
            # wait to get current position
            topic_address = '/' + self.arm.prefix + 'driver/out/joint_state'
            rospy.Subscriber(topic_address, JointState, self._set_joint_state)
            # the wait is blocking.
            rospy.wait_for_message(topic_address, JointState, timeout=10)
            self.req_submitted = True

    def _set_joint_state(self, sensor_msg):
        # _set_current_pose gets called each time the topic has a message.
        # So it will be running in the background once it gets called once.
        self._current_dof_pos, self._current_dof_vel, self._current_dof_torque = self._parse_msg(sensor_msg)

    def current_joint_state(self):
        def _format(sublist):
            # Convert each element to a string rounded to 3 decimal points
            rounded_values = [f"{value:.4f}" for value in sublist]
            return "[" + ",".join(rounded_values) + "]"

        self._get_joint_state()
        if self.capture_exec_traj:
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.dof_js_tracker.append((f"[{timestamp_str}]", _format(self._current_dof_pos),
                                        _format(self._current_dof_vel), _format(self._current_dof_torque)))
        return self._current_dof_pos, self._current_dof_vel, self._current_dof_torque

    def display_attempts(self):
        print("Robot DOF Angle, DOF Vel, DOF Torque")
        for _idx, ti_robot_js in enumerate(self.dof_js_tracker):
            result = str(ti_robot_js).replace("\n", " ")
            print(f"Action {_idx}: {result}\n")

        if self.capture_exec_traj:
            self.save_executed_dof_states()

    def save_executed_dof_states(self):
        file_path = get_expt_path(self.collision_js_path)
        # Open the file for writing
        with open(file_path, "w") as file:
            # Iterate over each sublist in the data
            for entry in self.dof_js_tracker:
                file.write(f"{entry}\n")


class JointStateSetter:
    """ Set either Joint Angles or Joint Velocities"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.target_dof_state = []
            cls._instance.dof_state_set_results = []
        return cls._instance

    def __init__(self, arm_, capture_cmd_traj):
        self.arm = arm_
        self.action_attempts = 0  # keep track of number of actions (write/set) performed so far.
        self.start_time = None

        # these paths are to capture the joint states (velocities or poses) when they are executed by RL
        self.capture_cmd_traj = capture_cmd_traj
        if capture_cmd_traj:
            collision_path_root = f'data/rosbags/external/kinova/rl/collisions/{_set}/inter/js'
            self.collision_js_path = os.path.join(notebooks_dir, collision_path_root)
            print(f"Capturing Trajectories at : {self.collision_js_path}.......")

    @staticmethod
    def rationalize_angles(unit, joint_value):
        """ Argument unit """

        if unit == 'degree':
            joint_degree_command = joint_value
            # get absolute value
            joint_degree_absolute_ = joint_degree_command
            joint_degree = joint_degree_absolute_
            joint_radian = list(map(math.radians, joint_degree_absolute_))
        elif unit == 'radian':
            assert all(x < 7 for x in joint_value), "Informal radian check failed."
            joint_degree_command = list(map(math.degrees, joint_value))
            joint_degree_absolute_ = joint_degree_command
            joint_degree = joint_degree_absolute_
            joint_radian = list(map(math.radians, joint_degree_absolute_))
        else:
            raise Exception("Joint value have to be in degree, or radian")

        return joint_degree, joint_radian

    def set_joint_angles(self, robot_dof_pos):
        if self.action_attempts == 0:
            self.start_time = time.time()

        try:
            self.target_dof_state.append(robot_dof_pos)
            target_position = [0] * 7  # the api takes 7 dofs only
            joint_degree, joint_radian = self.rationalize_angles(unit='radian', joint_value=robot_dof_pos)

            for _i in range(0, self.arm.arm_joint_number):
                target_position[_i] = joint_degree[_i]

            print(f"Attempting target : {target_position}")
            _run_result = joint_state_handler.joint_angle_client(arm=self.arm, angle_set=target_position)
            self.dof_state_set_results.append(_run_result)
            self.action_attempts += 1

        except rospy.ROSInterruptException as e:
            for _r_idx, _res in enumerate(self.dof_state_set_results):
                print(f"Run Status {_r_idx}: {_res}")
            raise ValueError(f"ROS interrupted before completion {str(e)}")

        elapsed_time = time.time() - self.start_time
        if self.action_attempts % 10 == 0:  # print elapsed time every 10 actions
            elapsed_time_p = round(elapsed_time, 3)
            freq_actions = round(self.action_attempts / elapsed_time, 3)
            print(f"Actions:{self.action_attempts}: Time elapsed: {elapsed_time_p} seconds, Freq: {freq_actions} Hz")

        return _run_result

    def set_joint_velocities(self, robot_dof_vel):

        if self.action_attempts == 0:
            self.start_time = time.time()
        try:
            self.target_dof_state.append(robot_dof_vel)
            target_velocities = [0] * 7  # the api takes 7 dofs only
            joint_degree, joint_radian = self.rationalize_angles(unit='radian', joint_value=robot_dof_vel)

            for _i in range(0, self.arm.arm_joint_number):
                target_velocities[_i] = joint_degree[_i]

            print(f"Attempting target vel: {target_velocities}")
            _run_result = joint_state_handler.joint_velocity_client(arm=self.arm, velocity_set=target_velocities)
            self.dof_state_set_results.append(_run_result)
            self.action_attempts += 1

        except rospy.ROSInterruptException as e:
            for _r_idx, _res in enumerate(self.dof_state_set_results):
                print(f"Run Status {_r_idx}: {_res}")
            raise ValueError(f"ROS interrupted before completion {str(e)}")

        elapsed_time = time.time() - self.start_time

        if self.action_attempts % 10 == 0:  # print elapsed time every 10 actions
            elapsed_time_p = round(elapsed_time, 3)
            freq_actions = round(self.action_attempts / elapsed_time, 3)
            print(f"Actions:{self.action_attempts}: Time elapsed: {elapsed_time_p} seconds, Freq: {freq_actions} Hz")

            if freq_actions < 30 or freq_actions > 100:
                print("Warning: Kinova expected joint vel freq is about 100Hz. Works fine till 50Hz")

        return _run_result

    def display_attempts(self):
        print("Robot DOF Angle/Velocity | Post DOF Pos Run Results")
        for _idx, (robot_pos, result) in enumerate(zip(self.target_dof_state, self.dof_state_set_results)):
            result = str(result).replace("\n", " ")
            # print(f"Action {_idx}: {robot_pos} | {result}\n")

        if self.capture_cmd_traj:
            self.save_attempted_dof_states()

    def save_attempted_dof_states(self):
        file_path = get_expt_path(self.collision_js_path)
        # Open the file for writing
        with open(file_path, "w") as file:
            # Iterate over each sublist in the data
            for sublist in self.target_dof_state:
                # Convert each element to a string rounded to 3 decimal points
                rounded_values = [f"{value:.4f}" for value in sublist]
                # Write the sublist with inner brackets to the file
                file.write("[" + ", ".join(rounded_values) + "]\n")


def get_expt_path(folder_path):
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder {folder_path} doesn't exist....")

    # Get list of files in the folder
    files = os.listdir(folder_path)

    # Filter out files that match the pattern 'expt_<number>.txt'
    existing_indices = []
    for file in files:
        if file.startswith('expt_') and file.endswith('.txt'):
            try:
                index = int(file.split('_')[1].split('.')[0])
                existing_indices.append(index)
            except ValueError:
                raise ValueError(f"Invalid filename found: {file}.")
        else:
            raise ValueError(f"Invalid filename found: {file}. Must start with 'expt_' and end with '.txt'.")

    # If no files match the pattern, return 'expt_0.txt'
    if not existing_indices:
        return os.path.join(folder_path, "expt_00.txt")

    # Otherwise, return the next index
    next_index = max(existing_indices) + 1
    return os.path.join(folder_path, f"expt_{next_index:02}.txt")
