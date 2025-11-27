import os
import random
import shutil


def random_file_from_folder(folder_path, file_type=None):
    """ Select a random file of a specified type from the given folder path. """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None  # Folder doesn't exist or is not a valid directory

    file_list = os.listdir(folder_path)
    if not file_list:
        return None  # Folder is empty

    if file_type:
        # Filter files based on the specified type
        file_list = [file for file in file_list if file.endswith(f".{file_type}")]

        if not file_list:
            return None  # No files of the specified type found

    random_file = random.choice(file_list)
    return random_file


def random_files_from_folder(folder_path, n=50):
    """Select random files from path."""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise ValueError("Folder doesn't exist or is not a valid directory")

    file_list = os.listdir(folder_path)
    if not file_list:
        raise ValueError("Folder is empty")

    if n > len(file_list):
        raise ValueError("Number of files to select (n) is greater than the number of files in the folder")

    random_files = random.sample(file_list, n)
    return random_files


def cleanup_image_location(image_dir):
    """ Check if the directory exists and if yes remove all images else create one. """
    if os.path.exists(image_dir):
        # If it exists, delete only the files with a ".png, jpg or npy" extension
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            try:
                if os.path.isfile(file_path) and (
                        filename.lower().endswith(".png")
                        or filename.lower().endswith(".jpg")
                        or filename.lower().endswith(".npy")
                ):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        # If it doesn't exist, create the directory
        try:
            os.makedirs(image_dir)
            print(f"Directory '{image_dir}' created.")
        except Exception as e:
            print(f"Error creating directory '{image_dir}': {e}")


def cleanup_location(_dir, file_types=(".png", ".jpg", ".npy")):
    """ Check if the directory exists and if yes remove all files of type else create one. """
    if os.path.exists(_dir):
        # If it exists, delete only the files with a ".png, jpg or npy" extension
        for filename in os.listdir(_dir):
            file_path = os.path.join(_dir, filename)
            try:
                for _type in file_types:
                    if os.path.isfile(file_path) and filename.lower().endswith(_type):
                        os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        # If it doesn't exist, create the directory
        try:
            os.makedirs(_dir)
            print(f"Directory '{_dir}' created.")
        except Exception as e:
            print(f"Error creating directory '{_dir}': {e}")


def cleanup_sub_folders_n_files(_dir, folder_prefix='careful', file_types=(".png", ".jpg", ".npy")):
    """ Check if the directory exists and if yes remove all files including subfolders. """
    if not os.path.exists(_dir) or not os.path.isdir(_dir):
        print(f"The folder '{_dir}' does not exist.")
        try:
            os.makedirs(_dir)
            print(f"Directory '{_dir}' created.")
        except Exception as e:
            print(f"Error creating directory '{_dir}': {e}")

        return

    assert '/spot/' in _dir and '/gen/' in _dir, "can only delete generated folders. careful there!"

    # Get the list of sub_folders
    sub_folders = [f.path for f in os.scandir(_dir) if f.is_dir()]

    for sf in sub_folders:
        # Check if sub_folder has the specified prefix
        if folder_prefix is not None and not os.path.basename(sf).startswith(folder_prefix):
            raise ValueError(f"Sub folder '{sf}' does not start with the specified prefix '{folder_prefix}'.")

    cleanup_location(_dir, file_types=file_types)  # delete files.
    for sf in sub_folders:
        try:
            shutil.rmtree(sf)

        except Exception as e:
            print(f"Failed to delete sub_folder '{sf}': {e}")
    print(f"Deleted sub folders of : {_dir}")


def count_files_in_folder(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)
        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        print(f"The folder '{folder_path}' was not found.")
        return None


def check_file_format(folder_path, _format='.png'):
    """Check if the folder exists and the file format """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Check each file for the '.png' extension
    for file in files:
        if not file.lower().endswith(_format):
            raise ValueError(f"{file} is not a {format} file.")


def count_sub_folders(folder_path):
    """Count the number of sub-folders in a folder"""
    try:
        # Get a list of all items in the folder
        items = os.listdir(folder_path)

        # Filter only subdirectories
        sub_folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]

        # Return the count of subdirectories
        return len(sub_folders)
    except Exception as e:
        print(f"Error counting sub_folders: {e}")
        return None


def copy_file(source_file, target_folder):
    """ Copy a file to a target folder."""
    assert '/spot/' in source_file, "can only copy spot files!"
    assert '/spot/' in target_folder, "can only copy to spot!"

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Use shutil.copy2 to copy the file along with its metadata
    shutil.copy2(source_file, target_folder)


def copy_folders(source_folder, target_folder):
    """ Copy from a parent folder to a target folder."""
    assert '/spot/' in source_folder, "can only copy from spot files!"
    assert '/spot/' in target_folder, "can only copy to spot!"
    # Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Iterate through each sub folder in the source folder
    for sub_folder in os.listdir(source_folder):
        source_sub_folder = os.path.join(source_folder, sub_folder)
        target_sub_folder = os.path.join(target_folder, sub_folder)

        # Check if the sub_folder is a directory
        if os.path.isdir(source_sub_folder):
            # Recursively copy the sub_folder and its contents
            shutil.copytree(source_sub_folder, target_sub_folder)


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def camel_to_snake(camel_str):
    snake_str = camel_str[0].lower()

    for char in camel_str[1:]:
        if char.isupper():
            snake_str += '_' + char.lower()
        else:
            snake_str += char
    return snake_str


def validate_file_types_in_sub_folders(folder_path, file_types):
    """
    Recursively checks if each sub folder in the given folder_path contains at least 1 file of each  file type.

    Args:
        folder_path (str): The path to the folder to check.
        file_types (list): A list of file extensions (e.g., [".txt", ".csv"]).

    Raises:
        ValueError: If a subfolder does not contain at least one file for each file type, with details.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path '{folder_path}' is not valid.")

    missing_files = {}

    for root, dirs, files in os.walk(folder_path):
        for sub_dir in dirs:
            sub_dir_path = os.path.join(root, sub_dir)
            sub_dir_files = [f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))]

            missing = []
            for file_type in file_types:
                if not any(f.endswith(file_type) for f in sub_dir_files):
                    missing.append(file_type)

            if missing:
                missing_files[sub_dir_path] = missing

    if missing_files:
        details = "\n".join([f"Folder: {folder}, Missing file types: {', '.join(types)}"
                             for folder, types in missing_files.items()])
        raise ValueError(f"Some subfolders are missing required file types:\n{details}")
