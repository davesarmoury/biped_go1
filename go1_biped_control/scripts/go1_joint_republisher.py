#!/usr/bin/env python3
import rospy
from unitree_legged_msgs.msg import LowState
from sensor_msgs.msg import JointState

deg2rad = 0.0174533

dof_names = ["RR_hip_joint",
            "RL_hip_joint",
            "FL_hip_joint",
            "FR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "RL_calf_joint",
            "FR_calf_joint",
            "RR_calf_joint"]

def low_callback(msg):
  global pub

  js_msg = JointState()
  js_msg.header.stamp = rospy.Time.now()
  js_msg.name = dof_names

  for m in msg.motorState:
    js_msg.position.append(m.q * deg2rad)

  pub.publish(js_msg)

def main():
    global pub
    rospy.init_node('go1_joints', anonymous=True)
    pub = rospy.Publisher('/joint_states', JointState, queue_size=10)
    rospy.Subscriber("/low_state", LowState, low_callback)
    rospy.spin()

main()
