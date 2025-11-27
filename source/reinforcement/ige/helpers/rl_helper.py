from isaacgym import gymapi
import random
import torch
import re
from estimation.math import common
from estimation.real.domain import ArmType
import yaml
import numpy as np
import hashlib
import json
from datetime import datetime


def arm_random_loc_within_reach(arm_base_pose, arm_type=ArmType.franka, _max_reach=None):
    """
    Generate a random location within reach for a robot arm in the front only.

    Args:
        arm_base_pose (tuple): The base location of the robot arm (x, y, z).
        arm_type : ArmType.franka or ArmType.kinova

    # Default Values:  Franka: j1_offset=0.3330, max_reach=0.855
    # Default Values:  Kinova: j1_offset=0, max_reach=0.90 (90 cm from dof1)
    Returns:
        tuple: A random location within reach (x, y, z).
    """
    if arm_type == ArmType.franka:
        j1_offset = 0.3330
        max_reach = 0.855
    elif arm_type == ArmType.kinova:
        j1_offset = 0.
        max_reach = 0.90  # 90 cm
    else:
        raise ValueError("Not Implemented yet.")

    if _max_reach is not None:
        max_reach = _max_reach

    x, y, z = arm_base_pose.p.x, arm_base_pose.p.y, arm_base_pose.p.z

    # add joint 1 offset in z axis (in franka, the reach is calculated from there.)
    z += j1_offset

    while True:
        random_x = random.uniform(-max_reach, 0)
        random_y = random.uniform(-max_reach, max_reach)
        random_z = random.uniform(0.1, max_reach)

        new_x = x + random_x
        new_y = y + random_y
        new_z = z + random_z

        distance = common.l2_norm(torch.Tensor([(x, y, z)]), torch.Tensor([(new_x, new_y, new_z)]))

        if distance <= max_reach:
            random_target_pose = gymapi.Transform()
            random_target_pose.p = gymapi.Vec3(new_x, new_y, new_z)
            random_target_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            return random_target_pose


def arm_random_loc_beyond_reach(arm_base_pose, arm_type=ArmType.kinova, max_reach=None):
    """
    Generate a random location just beyond the robot arm's reach but within a maximum reach distance.

    Args:
        arm_base_pose (tuple): The base location of the robot arm (x, y, z).
        arm_type : ArmType.franka or ArmType.kinova
        max_reach (float): The maximum distance for generating a random point. Must be > arm's own reach.

    Returns:
        gymapi.Transform: A random location just beyond arm reach (x, y, z).

    Raises:
        ValueError: If arm_type is unsupported or arm's reach is greater than or equal to max_reach.
    """
    if arm_type == ArmType.franka:
        j1_offset = 0.3330
        arm_reach = 0.855
    elif arm_type == ArmType.kinova:
        j1_offset = 0.
        arm_reach = 0.80
    else:
        raise ValueError("Unsupported arm type.")

    if max_reach is None:
        raise ValueError("max_reach must be specified.")
    if arm_reach >= max_reach:
        raise ValueError(f"arm_reach ({arm_reach}) must be less than max_reach ({max_reach})")

    x, y, z = arm_base_pose.p.x, arm_base_pose.p.y, arm_base_pose.p.z
    z += j1_offset  # Adjust for joint offset

    while True:
        # Generate a random point in front and slightly to the side
        random_x = random.uniform(-max_reach, 0)
        random_y = random.uniform(-max_reach, 0.2)
        random_z = random.uniform(0.1, max_reach)

        new_x = x + random_x
        new_y = y + random_y
        new_z = z + random_z

        distance = common.l2_norm(torch.Tensor([(x, y, z)]), torch.Tensor([(new_x, new_y, new_z)]))

        if arm_reach < distance <= max_reach:
            random_target_pose = gymapi.Transform()
            random_target_pose.p = gymapi.Vec3(new_x, new_y, new_z)
            random_target_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            return random_target_pose


def sort_kp(kp_tensor, num_envs, num_keypoints):
    """ sort key points in the order of its distance to the (0,0) pixel location"""
    assert kp_tensor.shape == torch.Size([num_envs, num_keypoints, 3])
    distances = torch.norm(kp_tensor[:, :, :2], dim=-1)
    sorted_indices = torch.argsort(distances, dim=1)

    # Apply the sorting to each 20x3 tensor
    sorted_kp = torch.gather(kp_tensor, 1, sorted_indices.unsqueeze(2).expand_as(kp_tensor))
    return sorted_kp


def get_tensor_storage_space(_tensor):
    """Get storage size utilized by a tensor"""
    # Get the element size (in bytes)
    element_size = _tensor.element_size()

    # Get the total number of elements in the tensor
    num_elements = _tensor.numel()

    # Calculate the total storage space consumed by the tensor in bytes
    total_storage_space_bytes = element_size * num_elements

    # Convert bytes to gigabytes
    total_storage_space_gb = total_storage_space_bytes / (1024 ** 3)

    return total_storage_space_gb


# Function to read the text file and create tensors
def tensors_from_file(file_path, device):
    """ Read a given txt file path and create a series of 1D tensors"""

    with open(file_path, 'r') as file:
        line_number = 0
        for line in file:
            # Skip lines that start with #
            if line.startswith('#'):
                continue

            line_number += 1

            # Split the line into individual values
            values = [float(x) for x in line.split(',')]

            yield line_number, torch.tensor(values).to(device)


def get_targets_by_line_num(file_path, device):
    """
    Returns: A dict of kvs, where each kv is (line_number, tensor from file).
    """

    tensor_dict = {}
    for line_number, tensor in tensors_from_file(file_path, device):
        tensor_dict[line_number] = tensor  # I think this is a 1D tensor of size 3
    return tensor_dict


def pos_and_quat_from_file_as_transforms(file_path):
    """
    Read a given txt file and create a list of gymapi.Transform objects
    from position and quaternion data.
    Input text format (ignores lines starting with '#'):
        [x, y, z], [qx, qy, qz, qw]

    Example:
        [0.731415, 0.160048, 0.196478], [-0.079670, 0.695131, 0.081353, 0.709808]
        [0.531415, 0.230480, 0.246478], [-0.079670, 0.695131, 0.081353, 0.709808]

    Output format:
        A list of gymapi.Transform() objects
    """
    transforms = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                # Expect two bracketed lists: position and quaternion
                pos_str, quat_str = line.split('], [')
                pos_str = pos_str.strip('[')
                quat_str = quat_str.strip(']')

                pos_vals = [float(x) for x in pos_str.split(',')]
                quat_vals = [float(x) for x in quat_str.split(',')]

                # Create Transform object
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(*pos_vals)
                transform.r = gymapi.Quat(*quat_vals)

                transforms.append(transform)
            except Exception as e:
                print(f"Failed to parse line: '{line}'. Error: {e}")

    return transforms


def extract_checkpoint_dir(checkpoint_path):
    """From 'runs/Test-16-10-08/nn/Franka.pth' extract the checkpoint dir 'Test-16-10-08'"""
    pattern = re.compile(r'runs/(.*?)/nn')

    # Use the pattern to find the match
    match = re.search(pattern, checkpoint_path)

    # Extract the word between "runs/" and "/nn" if there's a match
    if match:
        extracted_word = match.group(1)
        return extracted_word
    else:
        raise ValueError("Invalid Checkpoint passed")


def branch_level(dof_name):
    """ From dof name such as 'joint-branch-B356L4P351-to-branch-B359L5P356'
        extract level of child (5) """
    # Split the string using 'to' and take the second half
    child_br_name = dof_name.split('to')[1]
    # Use regex to find integers between 'L' and 'P'
    matches = re.findall(r'L(\d+)P', child_br_name)
    level = int(matches[0])
    assert 0 < level <= 7, f"level,{level}"
    return level


def child_branch_name(dof_name):
    """ From dof name like 'joint-branch-B356L4P351-to-branch-B359L5P356'
        extract name of child (branch-B359L5P356) """
    return dof_name.split('to-')[1]


def flatten_confs(d, parent_key='', sep='#', exclude_start='${'):
    """
    Flatten a nested dictionary.

    Parameters:
    - d: The input dictionary.
    - parent_key: The prefix to use for the keys in the flattened dictionary.
    - sep: The separator to use between keys.
    - exclude_start: ignore relative values from yaml

    Returns:
    A flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_confs(v, new_key, sep=sep).items())
        else:
            if not str(v).startswith(exclude_start):
                items.append((k, v))
    return dict(items)


# noinspection PyDefaultArgument
def match_train_test_confs(test_conf: dict, train_yaml_path: str,
                           ignores=(
                                   'numEnvs', 'name', 'episodeLength', 'contact_offset', 'rest_offset',
                                   'numObservations', 'pcShapeNormalizerDownSampleSize',
                                   'numActions', 'use_gpu_pipeline', 'use_gpu', 'testTreeUseFileRotation',
                                   'testLsystemAssetRoot', 'assetFileNameTree', 'maxTrgTorquesToStore'),
                           new_kvs={'transferableToReal': True,
                                    'dynamicsByBeamDeflection': False,
                                    'randomiseRobotStartPose': False,
                                    'enableSymmetryAwareness': False}):
    """ Ensure that the incoming test configs match the confs used for training, else throw an error.
        The keys in ignores are ignored, for e.g. the episode length can be different in train/test
        The new_keys, added during test can also be ignored. """
    try:
        # Load values from the YAML file
        with open(train_yaml_path, 'r') as yaml_file:
            train_conf = yaml.safe_load(yaml_file)
            flat_train_conf = flatten_confs(train_conf)

        assert train_conf['task']['name'] == test_conf['name'], 'Train/Test mismatch: Invalid Task Name'

        flat_test_conf = flatten_confs(test_conf)
        assert len(flat_test_conf) > 10 and len(flat_train_conf) > 10, "Incorrect Load of tst or train confs"

        # Iterate through elements in test_conf
        for key, value_test in flat_test_conf.items():

            if key not in ignores:
                # if keys are recently added, that was not in training, it should match teh default
                if key in new_kvs.keys() and key not in flat_train_conf:
                    value_default = new_kvs[key]
                    if value_test != value_default:  # Check if the values match
                        raise ValueError(
                            f"Default/Test mismatch: Key:'{key}'=> Test: '{value_test}', Default: '{value_default}'")
                else:
                    if key not in flat_train_conf:  # Check if the key exists in train_conf
                        raise ValueError(f"Train/Test mismatch: Key '{key}' not found in the YAML file")

                    value_train = flat_train_conf[key]
                    if value_test != value_train and key not in ignores:  # Check if the values match
                        raise ValueError(
                            f"Train/Test mismatch: Key:'{key}'=> Test: '{value_test}', Train: '{value_train}'")

    except Exception as e:
        raise ValueError(e)


def tensor_2d_range_info(_tensor, _tensor_name=''):
    """
    Given a tensor, like franka dof_pos or dof_vel computes and returns a dictionary containing
    min, max, and quantiles (0, 0.01, 0.5, 0.99).

    Args:
        _tensor: The input tensor.
        _tensor_name: Optional name for the tensor (for better readability).
    """

    assert _tensor.dim() == 2, "currently tested only for 2 dims, e.g. net_cf: [num_env x num_dofs]"

    if _tensor.numel() != 0:
        result_dict = {
            'min': torch.min(_tensor),
            'max': torch.max(_tensor),
            '0': torch.min(_tensor),  # 0th percentile is the same as min
            '0.01': torch.quantile(_tensor, 0.01),
            '0.50': torch.quantile(_tensor, 0.50),
            '0.99': torch.quantile(_tensor, 0.99)
        }

        print(f"{_tensor_name}[{_tensor.shape[1]}]: min: {result_dict['min']:.3f}, 1%ile: {result_dict['0.01']:.3f}, "
              f"50%ile: {result_dict['0.50']:.3f}, 99%ile: {result_dict['0.99']:.3f}, "
              f"max: {result_dict['max']:.3f}")
        return result_dict
    else:
        print("quantiles/min/max not available")
        return None


def validate_2d_obs_space(obs_tensor, obs_tensor_name=''):
    """ Validates that the 1st and 99th percentiles of the given tensor from the observation space (e.g. scaled_dof_pos)
    are within the range of -1 and 1.
    Loosely based on : https://forums.developer.nvidia.com/t/compute-observations-in-the-custom-task/178059/7 """

    quantile_dict = tensor_2d_range_info(obs_tensor, obs_tensor_name)

    if quantile_dict is not None:
        lq = quantile_dict['0.01']
        hq = quantile_dict['0.99']

        assert -1.01 <= lq and hq <= 1.01, f"{obs_tensor_name}: Invalid observation space: 1st percentile ({lq}) " \
                                           f"or 99th percentile ({hq}) is outside the range of -1 and 1."


def tensor_3d_range_info(_tensor, find_max_row=False):
    """ Given a tensor like franka net contact force, print the range (min, max, 1%ile, 99%ile)"""

    torch.set_printoptions(precision=2, sci_mode=False)
    assert _tensor.dim() == 3, "currently tested only for 3 dims, e.g. net_cf: [num_env x bodies x coord]"

    # # Find indices of non-zero elements
    nonzero_indices = torch.nonzero(_tensor, as_tuple=False)

    # Extract non-zero elements
    nonzero_elements = _tensor[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]

    if nonzero_elements.numel() != 0:
        lq = torch.quantile(nonzero_elements, 0.01)
        hq = torch.quantile(nonzero_elements, 0.99)
        print(f"min: {torch.min(_tensor)}, 1%ile: {lq}, 99%ile: {hq}, max: {torch.max(_tensor)}")
    else:
        print("quantiles/min/max not available")

    if find_max_row:
        # Find the flattened index of the maximum element
        flat_index = torch.argmax(_tensor).item()

        # Calculate the indices from the flattened index
        max_indices = torch.tensor(divmod(flat_index, _tensor.shape[1] * _tensor.shape[2]))

        # Extract the row index
        max_row = max_indices[0]
        print("Row of the Maximum Element:", max_row)


def rand_position_single_axis(low, high, seed=None):
    """ Generate random position along a single axis"""
    if seed is not None:
        random.seed(seed)
    assert low >= 0.1, "avoid collision with trunk"
    return round(random.uniform(low, high), 2)


def rand_n(n, rand_limit=5, seed=None):
    """ Generate n random values from -rand_limit, rand_limit"""
    if seed is not None:
        random.seed(seed)

    return [random.uniform(-rand_limit, rand_limit) for _ in range(n)]


def rotate_quat_by_degree(original_quat, rot_degree):
    """ Given an isaac gym quaternion {original_quat} rotate it by a {rot_degree} degree along z-axis
        to introduce randomness in tree rotation placement for domain Randomisation"""
    assert rot_degree <= 15, "too much rotation for the tree is dangerous. check the viewer to confirm"
    assert not isinstance(rot_degree, list), "The variable should not be a list"
    original_zyx = gymapi.Quat.to_euler_zyx(original_quat)

    rand_rot_radians = np.radians(rot_degree)
    updated_z_y_x = [original_zyx[0], original_zyx[1], original_zyx[2] + rand_rot_radians]
    rand_quat = gymapi.Quat.from_euler_zyx(*updated_z_y_x)
    return rand_quat


def rand_rotate(original_quat, rand_rot_limit_degree=5, seed=None):
    """ Given an isaac gym quaternion rotate it by a random degree along z-axis
        to introduce randomness in tree rotation placement for domain Randomisation"""

    if seed is not None:
        random.seed(seed)

    # assert rand_rot_limit_degree <= 5, "too much rotation for the tree is dangerous. check the viewer to confirm"
    original_zyx = gymapi.Quat.to_euler_zyx(original_quat)
    rand_rot_degrees = random.uniform(-1 * rand_rot_limit_degree, rand_rot_limit_degree)

    rand_rot_radians = np.radians(rand_rot_degrees)
    updated_z_y_x = [original_zyx[0], original_zyx[1], original_zyx[2] + rand_rot_radians]
    rand_quat = gymapi.Quat.from_euler_zyx(*updated_z_y_x)
    return rand_quat


def analyze_baseline_deviations(baseline_tensor, updated_tensor, deviation_thresholds=(1, 0.5)):
    """
    Compute L2 distances between baseline and updated tensors and analyze deviation distribution.

    Parameters:
    baseline_tensor (torch.Tensor): Baseline tensor of shape [N, 121, 3]
    updated_tensor (torch.Tensor): Updated tensor of shape [N, 121, 3]
    distance_thresholds ([float], optional): Threshold for sum of distances

    """
    assert baseline_tensor.shape == updated_tensor.shape, "Tensors must have the same shape"
    assert baseline_tensor.dim() == 3
    assert baseline_tensor.shape[2] == 3, "x-y-z should be present"

    # Compute point-wise L2 distances
    point_distances = torch.norm(baseline_tensor - updated_tensor, dim=2)

    # Sum distances across 120 points for each N
    sum_distances = torch.sum(point_distances, dim=1)

    for deviation_threshold in deviation_thresholds:
        # Calculate percentage of N where sum of distances > threshold
        above_threshold = sum_distances > deviation_threshold
        percent_above_threshold = (torch.sum(above_threshold).float() / len(sum_distances)) * 100
        print(f"pct of envs where deviation to baseline is above {deviation_threshold}m: {percent_above_threshold} %")


def create_dict_hash(dictionary):
    """ Create a hash-string of size 8 from a python dictionary"""
    # Sort the dictionary by keys to ensure consistent hashing
    sorted_dict = dict(sorted(dictionary.items()))

    dict_string = json.dumps(sorted_dict, sort_keys=True)

    # Create hash using SHA-256
    hash_object = hashlib.sha256(dict_string.encode())
    hash_string = hash_object.hexdigest()

    return hash_string[:8]


def transform_to_local_coordinates(A_positions, B_position, B_rotation):
    """
    Transforms world positions of n objects A1, A2, ..., An into the local coordinates of object B.

    Args:
        A_positions: Tensor of shape [num_envs, n, 3] containing world positions of A1, A2, ..., An.
        B_position: Tensor of shape [num_envs, 1, 3] containing world position of B.
        B_rotation: Tensor of shape [num_envs, 1, 4] containing quaternion [x, y, z, w] of B's rot.

    Returns:
        Tensor of shape [num_envs, n, 3] of A1, A2, ..., An positions in B's local coordinate frame.
    """

    # Verify input shapes
    assert len(A_positions.shape) == 3 and A_positions.shape[2] == 3, f"Ap Shape Error: {A_positions.shape}"
    assert len(B_position.shape) == 3 and B_position.shape[1] == 1 and B_position.shape[
        2] == 3, f"Bp Shape Error:  {B_position.shape}"
    assert len(B_rotation.shape) == 3 and B_rotation.shape[1] == 1 and B_rotation.shape[
        2] == 4, f"Br Shape Error:{B_rotation.shape}"

    # Additional check to ensure num_envs is consistent across all inputs
    num_envs = A_positions.shape[0]
    assert B_position.shape[0] == num_envs, f"Shape Error: {B_position.shape[0]} vs {num_envs}"
    assert B_rotation.shape[0] == num_envs, f"Shape Error: {B_rotation.shape[0]} vs {num_envs}"

    # Compute the relative positions
    relative_positions = A_positions - B_position  # Broadcasted subtraction [num_envs, 2, 3]

    # Normalize quaternion (if necessary)
    B_rotation = B_rotation / torch.linalg.norm(B_rotation, dim=-1,
                                                keepdim=True)  # [num_envs, 1, 4]

    # Compute the inverse quaternion of B
    B_rotation_inv = B_rotation.clone()
    B_rotation_inv[..., :3] *= -1  # Invert x, y, z parts (conjugate quaternion)

    def quaternion_rotate(q, v):
        """Rotates vector v by quaternion q."""
        q_xyz = q[..., :3]  # [x, y, z]
        q_w = q[..., 3:]  # [w]

        # Cross product: q_xyz x v
        cross1 = torch.cross(q_xyz, v, dim=-1)
        # Cross product: q_xyz x (q_xyz x v)
        cross2 = torch.cross(q_xyz, cross1, dim=-1)

        # Quaternion rotation formula
        rotated_v = v + 2 * (q_w * cross1 + cross2)
        return rotated_v

    # Rotate relative positions using the inverse quaternion
    local_positions = quaternion_rotate(B_rotation_inv, relative_positions)

    return local_positions


def point_cloud_viz(points, message='', overlay_points=None):
    """ Open 3D visualisation of {points}. Optional {overlay_points} to overlay. {message} displayed in window """
    import open3d as o3d

    if torch.is_tensor(points) and points.is_cuda:
        points = points.cpu()

    # Get screen dimensions
    import tkinter as tk
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Set window dimensions (half of the screen)
    window_width = screen_width // 2
    window_height = screen_height
    window_left = 0
    window_top = 0

    # Create main point cloud
    pcd = o3d.geometry.PointCloud()
    points_np = np.array(points)  # points should be of shape (N, 3)
    pcd.points = o3d.utility.Vector3dVector(points_np)

    if overlay_points is not None:
        pcd.paint_uniform_color([0, 0.1, 0.8])  # Red color for main points

    geometries = [pcd]

    # Add overlay points if provided
    if overlay_points is not None:
        if torch.is_tensor(overlay_points) and overlay_points.is_cuda:
            overlay_points = overlay_points.cpu()

        overlay_pcd = o3d.geometry.PointCloud()
        overlay_np = np.array(overlay_points)  # overlay_points should also be of shape (M, 3)
        overlay_pcd.points = o3d.utility.Vector3dVector(overlay_np)
        overlay_pcd.paint_uniform_color([1, 0, 0])  # Green color for overlay points
        geometries.append(overlay_pcd)

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"{message}",
        width=window_width,
        height=window_height,
        left=window_left,
        top=window_top,
    )

    # Add geometries to the visualizer
    for geom in geometries:
        vis.add_geometry(geom)

    # Add coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # Get the view control from the visualizer
    view_ctl = vis.get_view_control()

    # Look at the origin [0, 0, 0]
    lookat = np.array([0, 0, 0])

    # Distance from the camera to the lookat point (radius)
    r = 5.0

    # Convert spherical coordinates (theta, phi) to cartesian coordinates for camera position
    theta = 0.4
    phi = 0.707

    # Camera position in spherical coordinates
    cam_x = r * np.cos(phi) * np.cos(theta)
    cam_y = r * np.cos(phi) * np.sin(theta)
    cam_z = r * np.sin(phi)

    camera_position = np.array([cam_x, cam_y, cam_z])

    # Set camera parameters
    view_ctl.set_lookat(lookat)
    view_ctl.set_front(camera_position - lookat)
    view_ctl.set_up([0, 0, 1.0])  # Set the up direction (z-axis)
    view_ctl.set_zoom(1.0)  # Adjust zoom level if needed (1.0 is default)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

    print("point cloud viz complete..")


def vec3_to_np_array(vec3):
    return np.array([vec3.x, vec3.y, vec3.z])


def vec3_to_tensor(vec3, device):
    return torch.tensor([vec3.x, vec3.y, vec3.z], dtype=torch.float32).to(device)


def vec3s_to_tensor(vec3s, device):
    """  Converts a list of isaac gym vec3 instances to a tensor. """
    return torch.tensor(
        [[vec3.x, vec3.y, vec3.z] for vec3 in vec3s],
        dtype=torch.float32
    ).to(device)


def quat4_to_tensor(quat4s, device):
    """  Converts a list of isaac gym Quat instances to a tensor. """
    return torch.tensor(
        [[quat4.x, quat4.y, quat4.z, quat4.w] for quat4 in quat4s],
        dtype=torch.float32
    ).to(device)


def track_tree_dof_vel_breaches(frame_no, tree_dof_vel, velocity_breach_env_tracker, threshold=100):
    """ Track the tree dof breaches of velocity limits or rapid movements resembling rupture
    threshold: in rad/s for joint velocity.  """

    # tensor_2d_range_info(tree_dof_vel, "tree_dof_vel")

    assert tree_dof_vel.dim() == 2, f"Expecting shape num_envs x num_tree_dofs not {tree_dof_vel.shape}"
    breached_envs = ((tree_dof_vel > threshold) | (tree_dof_vel < -threshold)).any(dim=1)

    # Get indices of environments that breached the threshold
    breached_indices = breached_envs.nonzero(as_tuple=True)[0]

    # Update the tracker
    for env_id in breached_indices.tolist():
        if env_id not in velocity_breach_env_tracker:
            velocity_breach_env_tracker[env_id] = 0
        velocity_breach_env_tracker[env_id] += 1  # Increment breach count

    if frame_no % 100 == 0:
        sorted_tracker = sorted(velocity_breach_env_tracker.items(), key=lambda x: x[1], reverse=True)

        print(frame_no, ": Breach Count (Descending Order):", len(sorted_tracker))
        print("Env  :Breaches => \n  [" + ', '.join(
            f'{env_id}:{count}' for env_id, count in sorted_tracker) + "]")

    # assert frame_no <=600


def track_dof_breaches(frame_no, dof_metrics, breach_env_tracker, threshold, metric_name=''):
    """ Track the tree dof breaches of velocity/force limits or rapid movements resembling rupture
    threshold: in rad/s for joint velocity.  """

    assert dof_metrics.dim() == 2, f"Expecting shape num_envs x num_tree_dofs not {dof_metrics.shape}"
    breached_envs = ((dof_metrics > threshold) | (dof_metrics < -threshold)).any(dim=1)

    # Get indices of environments that breached the threshold
    breached_indices = breached_envs.nonzero(as_tuple=True)[0]

    # Update the tracker
    for env_id in breached_indices.tolist():
        if env_id not in breach_env_tracker:
            breach_env_tracker[env_id] = 0
        breach_env_tracker[env_id] += 1  # Increment breach count

    if frame_no % 100 == 0:
        sorted_tracker = sorted(breach_env_tracker.items(), key=lambda x: x[1], reverse=True)

        print(f"Frame: {frame_no}: {metric_name}: Breach Count (Descending Order):", len(sorted_tracker))
        print("Env:Breaches => \n  [" + ', '.join(
            f'{env_id}:{count}' for env_id, count in sorted_tracker) + "]")

    # assert frame_no <=600


def calculate_partial_success_rates(num_envs, overall_succ_mask, invalid_indices, device):
    """
    Group/Split success rates by valid and invalid indices.

    Args:
        overall_succ_mask (torch.Tensor): Boolean tensor indicating success/failure
        invalid_indices (list): List of indices to exclude from valid calculation
        num_envs (int, optional): Total number of environments. Defaults to length of succ_mask.
        device: device

    Returns:
        dict: Dictionary containing success rates and counts for valid and invalid subsets
    """

    # Create masks for valid and invalid indices
    valid_mask = torch.ones(num_envs, dtype=torch.bool).to(device)
    valid_mask[invalid_indices] = False
    invalid_mask = ~valid_mask

    # Filter success mask for valid and invalid environments
    valid_succ_mask = overall_succ_mask[valid_mask]
    invalid_succ_mask = overall_succ_mask[invalid_mask]

    # Calculate success counts
    valid_successes = valid_succ_mask.sum().item()
    invalid_successes = invalid_succ_mask.sum().item()

    # Calculate total counts
    num_valid = valid_mask.sum().item()
    num_invalid = invalid_mask.sum().item()

    # Calculate success rates
    valid_success_rate = valid_successes / num_valid if num_valid > 0 else 0
    invalid_success_rate = invalid_successes / num_invalid if num_invalid > 0 else 0
    overall_success_rate = overall_succ_mask.sum().item() / num_envs

    result = {
        "valid": {
            "success_rate": valid_success_rate,
            "successes": valid_successes,
            "total": num_valid
        },
        "invalid": {
            "success_rate": invalid_success_rate,
            "successes": invalid_successes,
            "total": num_invalid
        },
        "overall": {
            "success_rate": overall_success_rate,
            "successes": overall_succ_mask.sum().item(),
            "total": num_envs
        }
    }
    print(result)


def fmt_time(time_from_epoch_sec):
    return datetime.fromtimestamp(time_from_epoch_sec).strftime('%Y-%m-%d %H:%M:%S')


def fmt_diff(curr_time, start_time):
    formatted_time = datetime.fromtimestamp(curr_time).strftime('%Y-%m-%d %H:%M:%S')
    time_diff = round(curr_time - start_time, 4)
    return formatted_time, time_diff


if __name__ == '__main__':
    franka_start_position = gymapi.Vec3(1.0, 0.0, 0.0)
    arm_start_pose = gymapi.Transform()
    arm_start_pose.p = franka_start_position
    arm_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

    for _ in range(70):
        voxel_random_pose = arm_random_loc_within_reach(arm_base_pose=arm_start_pose, arm_type=ArmType.kinova)
        print(round(voxel_random_pose.p.x, 3), round(voxel_random_pose.p.y, 3), round(voxel_random_pose.p.z, 3),
              sep=',')
