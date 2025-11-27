# Run custom isaacgymenvs tasks by copying it to the original repository.
#!/bin/bash

set -e

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Usage: $0 task=<task_name>"
  exit 1
fi



# Check if the argument is in the "key=value" format
if [[ "$1" == *"task="* ]]; then
  # Extract the value part after the equal sign
  input="$1"
  camel_case_task="${input#*=}"
else
  # If the argument is not in "key=value" format, throw an error
  echo "Error: Expected an argument with the format task=<task_name>"
  exit 1
fi

# Convert CamelCase to snake_case
snake_case_task=$(echo "$camel_case_task" | sed -r 's/([a-z0-9])([A-Z])/\1_\L\2/g')
snake_case_task=$(echo "$snake_case_task" | tr '[:upper:]' '[:lower:]')

echo "snake_case_task: $snake_case_task"


# Define the string to search for
task_name="$camel_case_task"


if [ -z "$isaacgymenvs_path" ]
then
  echo "Please set isaacgymenvs_path first. Refer readme.cmd"
  exit 0
else
  echo "isaacgymenvs_path = $isaacgymenvs_path"
fi

echo "Installing custom isaacgymenvs RL packages to $isaacgymenvs_path"

echo "removing task files from $isaacgymenvs_path ..... "

rm -f $isaacgymenvs_path/tasks/$snake_case_task.py
rm -f $isaacgymenvs_path/cfg/task/$camel_case_task.yaml
rm -f $isaacgymenvs_path/cfg/train/${camel_case_task}PPO.yaml

echo "copying task files ...."
yes | cp -f $spot_path/source/reinforcement/ige/tasks/$snake_case_task.py $isaacgymenvs_path/tasks/
yes | cp -f $spot_path/source/reinforcement/ige/cfg/task/$camel_case_task.yaml $isaacgymenvs_path/cfg/task/
yes | cp -f $spot_path/source/reinforcement/ige/cfg/train/${camel_case_task}PPO.yaml $isaacgymenvs_path/cfg/train/
echo "Copy Code: $? - Successful"

# Take a backup of all relevant task files.
curr_timestamp=$(date +"%Y-%m-%d-%H-%M-%S")

is_test="false"

# Loop through all arguments
for arg in "$@"; do
  if [ "$arg" = "test=True" ]; then
    is_test="true"
  fi
done

if [ "$is_test" == "false" ]; then
    task_backup_folder="$spot_path/notebooks/work2/data/tactile/simulation/rl/tasks/task_train_$curr_timestamp"
    mkdir "$task_backup_folder"
    echo "Task Backup Folder created: $task_backup_folder"

    cp -f $spot_path/source/reinforcement/ige/tasks/$snake_case_task.py $task_backup_folder/
    cp -f $spot_path/source/reinforcement/ige/cfg/task/$camel_case_task.yaml $task_backup_folder/
    cp -f $spot_path/source/reinforcement/ige/cfg/train/${camel_case_task}PPO.yaml $task_backup_folder/
fi

# Define the file to search in
file_to_search="$isaacgymenvs_path/tasks/__init__.py"  # Replace with the actual file path

# Check if the string is present in the file using grep
if grep -q "$task_name" "$file_to_search"; then
    echo "Task '$task_name' found in isaacgym_task_map from '$file_to_search'."
else
    echo "Error: Task '$task_name' not found in isaacgym_task_map. Add entry in '$file_to_search'"
    exit 1
fi

echo "Ready to Run RL .. "


# Use the latest run checkpoint path automatically if test=True and if checkpoint is not already specified
latest_folder_flag=false
# Iterate through the arguments
for arg in "$@"; do
  if [ "$arg" = "test=True" ]; then
    latest_folder_flag=true
  fi
  # You can add additional argument checks here as needed.
done


for arg in "$@"; do
if [[ "$arg" == *"checkpoint="* ]]; then
    latest_folder_flag=false
  fi
  # You can add additional argument checks here as needed.
done


runs_dir="runs"

# Check if the directory exists
if [ -d "$runs_dir" ] && [ "$latest_folder_flag" = true ]; then
  # Use 'ls' to list directories in the given directory, sorted by modification time
  # -t: Sort by modification time (newest first)
  # -d: List directories only (not their contents)
  # |: Pipe the output to 'tail' to get the last line (latest directory)
  latest_run_folder=$(ls -td "$runs_dir"/${camel_case_task}* | head -n 1)

  if [ -n "$latest_run_folder" ]; then
    echo "The latest folder in '$runs_dir' is: $latest_run_folder"
  else
    echo "No folders found in '$runs_dir'"
  fi
else
  echo "Skipping search in '$runs_dir'or it does not exist."
fi


if [ "$latest_folder_flag" = true ] ; then
  final_command="HYDRA_FULL_ERROR=1 python $isaacgymenvs_path/train.py $@ checkpoint=${latest_run_folder}/nn/${camel_case_task}.pth"
else
  final_command="HYDRA_FULL_ERROR=1 python $isaacgymenvs_path/train.py $@"
fi

# pass arguments to the script as is to the python train script.
echo "$final_command"
eval "$final_command"
