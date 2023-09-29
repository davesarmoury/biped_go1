#!/usr/bin/env python3

## This whole script feels dirty, but it should work

import rospy
from unitree_legged_msgs.msg import MotorState, MotorCmd, LowState, LowCmd

states = [MotorState] * 12

def FL_calf_callback(msg):
   states[5] = msg

def FL_hip_callback(msg):
   states[3] = msg

def FL_thigh_callback(msg):
   states[4] = msg

def FR_calf_callback(msg):
   states[2] = msg

def FR_hip_callback(msg):
   states[0] = msg

def FR_thigh_callback(msg):
   states[1] = msg

def RL_calf_callback(msg):
   states[11] = msg

def RL_hip_callback(msg):
   states[9] = msg

def RL_thigh_callback(msg):
   states[10] = msg

def RR_calf_callback(msg):
   states[8] = msg

def RR_hip_callback(msg):
   states[6] = msg

def RR_thigh_callback(msg):
   states[7] = msg

def low_callback(msg):
  global FL_calf_pub, FL_hip_pub, FL_thigh_pub ,FR_calf_pub, FR_hip_pub, FR_thigh_pub ,RL_calf_pub, RL_hip_pub, RL_thigh_pub, RR_calf_pub, RR_hip_pub, RR_thigh_pub
  rospy.loginfo("low")
  FL_calf_pub.publish(msg.motorCmd[5])
  FL_hip_pub.publish(msg.motorCmd[3])
  FL_thigh_pub.publish(msg.motorCmd[4])
  FR_calf_pub.publish(msg.motorCmd[2])
  FR_hip_pub.publish(msg.motorCmd[0])
  FR_thigh_pub.publish(msg.motorCmd[1])
  RL_calf_pub.publish(msg.motorCmd[11])
  RL_hip_pub.publish(msg.motorCmd[9])
  RL_thigh_pub.publish(msg.motorCmd[10])
  RR_calf_pub.publish(msg.motorCmd[8])
  RR_hip_pub.publish(msg.motorCmd[6])
  RR_thigh_pub.publish(msg.motorCmd[7])

def main():
    global state_pub
    global FL_calf_pub, FL_hip_pub, FL_thigh_pub ,FR_calf_pub, FR_hip_pub, FR_thigh_pub ,RL_calf_pub, RL_hip_pub, RL_thigh_pub, RR_calf_pub, RR_hip_pub, RR_thigh_pub

    rospy.init_node("go1_sim_macguffin", anonymous=True)
    state_pub = rospy.Publisher("/low_state", LowState, queue_size=10)

    rospy.Subscriber("/go1_gazebo/FL_calf_controller/state", MotorState, FL_calf_callback)
    rospy.Subscriber("/go1_gazebo/FL_hip_controller/state", MotorState, FL_hip_callback)
    rospy.Subscriber("/go1_gazebo/FL_thigh_controller/state", MotorState, FL_thigh_callback)
    rospy.Subscriber("/go1_gazebo/FR_calf_controller/state", MotorState, FR_calf_callback)
    rospy.Subscriber("/go1_gazebo/FR_hip_controller/state", MotorState, FR_hip_callback)
    rospy.Subscriber("/go1_gazebo/FR_thigh_controller/state", MotorState, FR_thigh_callback)
    rospy.Subscriber("/go1_gazebo/RL_calf_controller/state", MotorState, RL_calf_callback)
    rospy.Subscriber("/go1_gazebo/RL_hip_controller/state", MotorState, RL_hip_callback)
    rospy.Subscriber("/go1_gazebo/RL_thigh_controller/state", MotorState, RL_thigh_callback)
    rospy.Subscriber("/go1_gazebo/RR_calf_controller/state", MotorState, RR_calf_callback)
    rospy.Subscriber("/go1_gazebo/RR_hip_controller/state", MotorState, RR_hip_callback)
    rospy.Subscriber("/go1_gazebo/RR_thigh_controller/state", MotorState, RR_thigh_callback)

    FL_calf_pub = rospy.Publisher("/go1_gazebo/FL_calf_controller/command", MotorCmd, queue_size=1)
    FL_hip_pub = rospy.Publisher("/go1_gazebo/FL_hip_controller/command", MotorCmd, queue_size=1)
    FL_thigh_pub = rospy.Publisher("/go1_gazebo/FL_thigh_controller/command", MotorCmd, queue_size=1)
    FR_calf_pub = rospy.Publisher("/go1_gazebo/FR_calf_controller/command", MotorCmd, queue_size=1)
    FR_hip_pub = rospy.Publisher("/go1_gazebo/FR_hip_controller/command", MotorCmd, queue_size=1)
    FR_thigh_pub = rospy.Publisher("/go1_gazebo/FR_thigh_controller/command", MotorCmd, queue_size=1)
    RL_calf_pub = rospy.Publisher("/go1_gazebo/RL_calf_controller/command", MotorCmd, queue_size=1)
    RL_hip_pub = rospy.Publisher("/go1_gazebo/RL_hip_controller/command", MotorCmd, queue_size=1)
    RL_thigh_pub = rospy.Publisher("/go1_gazebo/RL_thigh_controller/command", MotorCmd, queue_size=1)
    RR_calf_pub = rospy.Publisher("/go1_gazebo/RR_calf_controller/command", MotorCmd, queue_size=1)
    RR_hip_pub = rospy.Publisher("/go1_gazebo/RR_hip_controller/command", MotorCmd, queue_size=1)
    RR_thigh_pub = rospy.Publisher("/go1_gazebo/RR_thigh_controller/command", MotorCmd, queue_size=1)

    rospy.Subscriber("/low_cmd", LowCmd, low_callback)

    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        state_msg = LowState()

        for index, state in enumerate(states):
          state_msg.motorState[index] = state
        
        state_pub.publish(state_msg)
        rate.sleep()

main()
