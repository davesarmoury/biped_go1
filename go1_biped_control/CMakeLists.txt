cmake_minimum_required(VERSION 3.0.2)
project(go1_biped_control)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES go1_biped_control
#  CATKIN_DEPENDS robot_localization
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
