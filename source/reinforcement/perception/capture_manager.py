import os
import numpy as np
from isaacgym import gymapi
from PIL import Image as im


def store_img_from_cpu(gym_state, image_dir, camera_handles, step_index, depth_capture=False):
    """ Capture and save images to disk.
        Warning: This function will generate images on cpu, hence will not be performant. """

    for i in range(gym_state.num_envs):
        # The gym utility to write images to disk is recommended only for RGB images.

        rgb_filename = f"{image_dir}/rgb_env{i}_img{step_index}.png"

        gym_state.gym.write_camera_image_to_file(
            gym_state.sim, gym_state.envs[i], camera_handles[i], gymapi.IMAGE_COLOR, rgb_filename
        )

        if depth_capture:
            # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
            # Here we retrieve a depth image, normalize it to be visible in an
            # output image and then write it to disk using Pillow
            depth_image = gym_state.gym.get_camera_image(
                gym_state.sim, gym_state.envs[i], camera_handles[i], gymapi.IMAGE_DEPTH
            )

            # -inf implies no depth value, set it to zero. output will be black.
            depth_image[depth_image == -np.inf] = 0

            # clamp depth image to 10 meters to make output image human friendly
            depth_image[depth_image < -10] = -10

            # flip the direction so near-objects are light and far objects are dark
            normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

            # Convert to a pillow image and write it to disk
            normalized_depth_image = im.fromarray(
                normalized_depth.astype(np.uint8), mode="L"
            )

            normalized_depth_image.save(f"{image_dir}/depth_env{i}_cam{step_index}.jpg")


def cleanup_image_location(image_dir, delete_sub_dirs=False):
    """ Check if the directory exists and if yes remove all images else create one. """
    if os.path.exists(image_dir):
        # If it exists, delete only the files with a ".png" extension
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            try:
                if os.path.isfile(file_path) and (
                        filename.lower().endswith(".png")
                        or filename.lower().endswith(".jpg")
                        or filename.lower().endswith(".npy")
                        or filename.lower().endswith(".pt")
                        or filename.lower().endswith(".pth")
                ):
                    os.unlink(file_path)

                if delete_sub_dirs is True and os.path.isdir(file_path):
                    last_folder = os.path.basename(file_path)
                    if last_folder.isdigit():
                        cleanup_image_location(file_path, delete_sub_dirs=False)

            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        # If it doesn't exist, create the directory
        try:
            os.makedirs(image_dir)
            print(f"Directory '{image_dir}' created.")
        except Exception as e:
            print(f"Error creating directory '{image_dir}': {e}")
