cmake_minimum_required(VERSION 3.0.2)
project(spot_control)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  kinova_driver
  kinova_msgs
  std_msgs
  geometry_msgs
  roscpp
  rospy
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES spot_control
  CATKIN_DEPENDS kinova_driver kinova_msgs roscpp rospy std_msgs geometry_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
install(PROGRAMS
  nodes/spot_control/gt_trajectory_tree_force_control.py
  nodes/spot_control/gt_trajectory_box_pos_control.py
  nodes/spot_control/gt_trajectory_tree_pos_control.py
  nodes/spot_control/gt_trajectory_tree_rl_ee_control.py
  nodes/spot_control/gt_trajectory_tree_ja_control.py
  nodes/spot_control/gt_trajectory_tree_jv_control.py
  nodes/spot_control/tactile_rl_txn_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
