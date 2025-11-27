#!/bin/bash

set -e

if [ -z "$spot_path" ]
then
  echo "Please set spot_path first. Refer readme.cmd"
  exit 0
else
  echo "spot_path = $spot_path"
fi

if [ -z "$kinova_ros_path" ]
then
  echo "Please set kinova_ros_path first. Refer readme.cmd"
  exit 0
else
  echo "kinova_ros_path = $kinova_ros_path"
fi

echo "Installing kinova spot packages to $kinova_ros_path"

echo "removing spot_* files from $kinova_ros_path ..... "
rm -rf $kinova_ros_path/spot_*


echo "copying kinova spot files ...."
yes | cp -rf $spot_path/external/kinova/spot_* $kinova_ros_path/
echo "Copy Code: $? - Successful"


echo "Ready to execute Jaco Commands .. "
