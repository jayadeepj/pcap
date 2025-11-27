""" settings, feature engineering fns, neural network architecture to perform kinova collision classification"""

import pandas as pd
import pytz
import torch

store_frame_lag = 10  # compute lag features for {store_frame_lag} past frames (so 1sec for kinova)
num_kinova_dofs = 6
roll_window_size = 2  # for noise minimisation, compute features over rolling window
assert store_frame_lag % roll_window_size == 0, "invalid setting for rolling window size"

root_feature_count = num_kinova_dofs  # the root features

# num kinova dofs x 1 each for var, min, max, skewness, kurtosis
total_torque_feature_cnt = root_feature_count + (num_kinova_dofs * 5) + (num_kinova_dofs * 5)
total_feature_cnt = total_torque_feature_cnt + (num_kinova_dofs * 2)  # add vel features (cmd and observed)


# Feature Engineering functions

def repeat_rows_up_to_limit(_tensor, n_lags):
    """ Repeat the first row of the input tensor, to create a [n_lags x ?] tensor. """
    if _tensor.size(0) >= n_lags:
        return _tensor

    # Repeat the first row to reach the specified limit
    repeated_rows = _tensor[:1].repeat((n_lags - _tensor.size(0)), 1)
    repeated_tensor = torch.cat([repeated_rows, _tensor], dim=0)

    return repeated_tensor


def create_lag_features_by_row(_tensor, n_lags, device):
    """Raw tensor, n_lags: no of steps to look back"""
    lagged_t = []

    assert _tensor.dim() == 2, "Invalid Shape"

    num_rows = _tensor.shape[0]
    assert 1 <= num_rows <= n_lags, f"The incoming rows must have length <= : {n_lags}"

    rep_tensor = repeat_rows_up_to_limit(_tensor, n_lags=n_lags)

    for i in range(n_lags - 1, len(rep_tensor)):
        lagged_row = []
        for j in range(0, n_lags):
            lagged_row.append([])
            lagged_row[-1] = rep_tensor[i - j].tolist()

        lagged_t.append(lagged_row)

    lag_features = torch.tensor(lagged_t).to(device)
    assert lag_features.shape == (rep_tensor.shape[0] - n_lags + 1,
                                  n_lags, num_kinova_dofs), f"Invalid shape: {lag_features.shape}"

    return lag_features


def build_features_by_window(dof_torque_tw, n_lags, device):
    """ For a window of incoming records (size = 1:10 x 6), build features.
        Input should the last 10 records from the trajectory stream. """

    assert dof_torque_tw.dim() == 2, "invalid shape"
    assert dof_torque_tw.shape[1] == num_kinova_dofs, "invalid shape"
    assert 1 <= dof_torque_tw.shape[0] <= n_lags, "invalid shape"

    dof_torque_features_w = dof_torque_tw[-1, :].unsqueeze(0).detach().clone()

    # Create lag features
    dof_torque_lag_comb_w = create_lag_features_by_row(dof_torque_tw, n_lags=n_lags, device=device)

    lag_variance_feature = torch.var(dof_torque_lag_comb_w, dim=1)
    lag_min_feature, _ = torch.min(dof_torque_lag_comb_w, dim=1)
    lag_max_feature, _ = torch.max(dof_torque_lag_comb_w, dim=1)

    mean = torch.mean(dof_torque_lag_comb_w, dim=1)
    std = torch.std(dof_torque_lag_comb_w, dim=1) + 1e-9
    lag_skewness_feature = torch.mean((dof_torque_lag_comb_w - mean[:, None]) ** 3 / std[:, None] ** 3, dim=1)
    lag_kurtosis_feature = torch.mean((dof_torque_lag_comb_w - mean[:, None]) ** 4 / std[:, None] ** 4, dim=1) - 3

    assert lag_variance_feature.shape == dof_torque_features_w.shape
    assert lag_min_feature.shape == dof_torque_features_w.shape
    assert lag_min_feature.shape == dof_torque_features_w.shape

    assert lag_skewness_feature.shape == dof_torque_features_w.shape
    assert lag_kurtosis_feature.shape == dof_torque_features_w.shape

    dof_torque_features_w = torch.cat((dof_torque_features_w, lag_variance_feature), dim=-1)
    dof_torque_features_w = torch.cat((dof_torque_features_w, lag_min_feature), dim=-1)
    dof_torque_features_w = torch.cat((dof_torque_features_w, lag_max_feature), dim=-1)

    dof_torque_features_w = torch.cat((dof_torque_features_w, lag_skewness_feature), dim=-1)
    dof_torque_features_w = torch.cat((dof_torque_features_w, lag_kurtosis_feature), dim=-1)

    # compute rolling features.
    # Reshape the tensor from [n, 10, 6 ] to [n, 5, 2, 6 ]
    reshaped_dof_torque_lag_comb_w = dof_torque_lag_comb_w.detach().clone().view(
        dof_torque_lag_comb_w.shape[0], -1, roll_window_size, dof_torque_lag_comb_w.shape[-1])

    roll_mean_features_w = torch.mean(reshaped_dof_torque_lag_comb_w, dim=2)

    roll_lag_variance_feature = torch.var(roll_mean_features_w, dim=1)
    roll_lag_min_feature, _ = torch.min(roll_mean_features_w, dim=1)
    roll_lag_max_feature, _ = torch.max(roll_mean_features_w, dim=1)

    dof_torque_features_w = torch.cat((dof_torque_features_w, roll_lag_variance_feature), dim=-1)
    dof_torque_features_w = torch.cat((dof_torque_features_w, roll_lag_min_feature), dim=-1)
    dof_torque_features_w = torch.cat((dof_torque_features_w, roll_lag_max_feature), dim=-1)

    mean = torch.mean(roll_mean_features_w, dim=1)
    std = torch.std(roll_mean_features_w, dim=1) + 1e-9
    roll_lag_skewness_feature = torch.mean((roll_mean_features_w - mean[:, None]) ** 3 / std[:, None] ** 3, dim=1)
    roll_lag_kurtosis_feature = torch.mean((roll_mean_features_w - mean[:, None]) ** 4 / std[:, None] ** 4, dim=1) - 3

    dof_torque_features_w = torch.cat((dof_torque_features_w, roll_lag_skewness_feature), dim=-1)
    dof_torque_features_w = torch.cat((dof_torque_features_w, roll_lag_kurtosis_feature), dim=-1)

    assert dof_torque_features_w.shape == (1, total_torque_feature_cnt), f"Invalid shape: {dof_torque_features_w.shape}"

    return dof_torque_features_w


def standardize_time(joint_torques_df):
    """ Feature Engineering fn to add aest time to in dataframe"""

    joint_torques_df = joint_torques_df.rename(columns={'Time': 'time'})
    joint_torques_df.sort_values(by=['time'], inplace=True)

    australian_time = pytz.timezone('Australia/Sydney')
    joint_torques_df['dt'] = pd.to_datetime(joint_torques_df['time'], unit='s', utc=True)
    joint_torques_df['fmt_dt'] = joint_torques_df['dt'].dt.tz_convert(australian_time)

    joint_torques_df['fmt_dt_aest'] = joint_torques_df['fmt_dt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    joint_torques_df['fmt_dt_aest'] = pd.to_datetime(joint_torques_df['fmt_dt_aest'])
    joint_torques_df.drop(columns=['dt', 'fmt_dt'], inplace=True)
    return joint_torques_df
