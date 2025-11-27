""" Interface to the real robot. functions in this file are invoked during sim2real """

import json
import requests
import torch
from reinforcement.ige.real.kinova import robot_collision_checker as cc
from reinforcement.ige.real.kinova import robot_domain
from reinforcement.ige.real.kinova import cartesian_robot_domain

from enum import Enum


class CommStatus(Enum):
    SUCCESS = 1
    ERROR = 2
    INVALID = 3


def invoke_kinova_post_service_ja(robot_dof_pos):
    """ Service to set joint angles in kinova"""
    url = 'http://172.17.34.245:3738/set_dof_pos'
    headers = {'Content-Type': 'application/json'}  # Set the Content-Type header
    payload = json.dumps({'robot_dof_pos': robot_dof_pos})
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return CommStatus.SUCCESS
    else:
        print("Failed to set joint angles:", response.text)
        return CommStatus.ERROR


def invoke_kinova_post_service_jv(robot_dof_vel):
    """ Service to set joint velocities in kinova"""
    url = 'http://172.17.34.245:3738/set_dof_vel'
    headers = {'Content-Type': 'application/json'}  # Set the Content-Type header
    payload = json.dumps({'robot_dof_vel': robot_dof_vel})
    response = requests.post(url, data=payload, headers=headers)
    if response.status_code == 200:
        return CommStatus.SUCCESS
    else:
        print("Failed to set joint velocities:", response.text)
        return CommStatus.ERROR


def invoke_kinova_fetch_service():
    url = 'http://172.17.34.245:3738/fetch_kinova_metrics'
    response = requests.get(url)
    if response.status_code == 200:
        return CommStatus.SUCCESS, response.json()
    else:
        print("Failed to fetch Kinova metrics:", response.text)
        return CommStatus.ERROR, response.text


def shutdown_kinova_service():
    url = 'http://172.17.34.245:3738/shutdown'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return CommStatus.SUCCESS, response.json()
        else:
            print("Failed to shutdown:", response.text)
            return CommStatus.ERROR, response.text
    except Exception as e:
        print("An error occurred during shutdown:", e)
        return CommStatus.ERROR, str(e)


def fetch_all_real_tensors(device, transform_to_sim=True, include_cartesian=False):
    """ Single shot fetch of all required topics from ROS"""

    def _json_resp_to_km(json_data):
        robot_dof_pos = json_data['robot_dof_pos']
        robot_dof_vel = json_data['robot_dof_vel']
        robot_dof_torque = json_data['robot_dof_torque']
        hand_pos = json_data['hand_pos']
        hand_rot = json_data['hand_rot']

        if include_cartesian:
            robot_link_cart_pos = json_data['robot_link_cart_pos']
            robot_link_cart_rot = json_data['robot_link_cart_rot']
            _kinova_metrics = cartesian_robot_domain.KinovaMetricsR(
                robot_dof_pos, robot_dof_vel, robot_dof_torque, hand_pos, hand_rot,
                robot_link_cart_pos, robot_link_cart_rot, transform_to_sim=transform_to_sim)
        else:
            _kinova_metrics = robot_domain.KinovaMetricsR(
                robot_dof_pos, robot_dof_vel, robot_dof_torque, hand_pos, hand_rot, transform_to_sim=transform_to_sim)

        return _kinova_metrics

    # Example usage
    status, km_json = invoke_kinova_fetch_service()
    assert status == CommStatus.SUCCESS, "Invalid response on fetching kinova metrics"
    kinova_metrics_r = _json_resp_to_km(km_json)

    metricsT = cartesian_robot_domain.KinovaMetricsT if include_cartesian else robot_domain.KinovaMetricsT
    kinova_metrics_t = metricsT.to_tensor(kinova_metrics_r, device)
    return kinova_metrics_t


def set_real_robot_dof_pos(actions):
    """ set 6 dof positions from kinova arm. called on pre_physics_step"""
    target_robot_dof_pos = actions.tolist()
    resp = invoke_kinova_post_service_ja(robot_dof_pos=target_robot_dof_pos)
    return resp


def set_real_robot_dof_vel(actions):
    """ set 6 dof velocities from kinova arm. called on pre_physics_step"""
    target_robot_dof_vel = actions.tolist()
    resp = invoke_kinova_post_service_jv(robot_dof_vel=target_robot_dof_vel)
    return resp


def real_collision_prob(loaded_model, classifier_type, curr_dof_torque_mem, obs_vel_features, sliding_cmd_vel_features):
    """ prob of real collision estimated by the classifier b/w 0 & 1  """

    assert obs_vel_features.shape == torch.Size([1, cc.num_kinova_dofs])
    assert sliding_cmd_vel_features.shape == torch.Size([1, cc.num_kinova_dofs])
    curr_dof_torque_t_win = curr_dof_torque_mem[-cc.store_frame_lag:, :]  # the window to perform feature engineering
    dof_torque_features_w = cc.build_features_by_window(curr_dof_torque_t_win,
                                                        n_lags=cc.store_frame_lag,
                                                        device=curr_dof_torque_mem.device)
    dof_vel_t = torch.cat((obs_vel_features, sliding_cmd_vel_features), dim=1)  # don't change the order.
    dof_vel_torque_features = torch.cat((dof_vel_t, dof_torque_features_w), dim=1)

    pred_probs, pred_labels = classifier_type.pred_labels(model=loaded_model,
                                                          x_features=dof_vel_torque_features,
                                                          prob_threshold=0.5)
    return pred_probs[-1, :], pred_labels[-1, :]


if __name__ == '__main__':
    # Example usage:

    kinova_metrics = fetch_all_real_tensors(device='cuda', transform_to_sim=True,
                                            include_cartesian=True)

    hand_pos = kinova_metrics.hand_pos_t
    hand_rot = kinova_metrics.hand_rot_t
    robot_dof_pos_t = kinova_metrics.robot_dof_pos_t

    print("hand_pos", hand_pos)
    print("hand_rot", hand_rot)
    print("robot_dof_pos_t", robot_dof_pos_t)
