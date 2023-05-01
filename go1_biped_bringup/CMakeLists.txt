cmake_minimum_required(VERSION 3.0.2)
project(go1_biped_bringup)

find_package(catkin REQUIRED COMPONENTS)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES go1_biped_bringup
#  CATKIN_DEPENDS go1_biped_description microstrain_inertial_driver
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
