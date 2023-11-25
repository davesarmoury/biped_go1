#!/usr/bin/env python3
import rospy
import torch
import threading
from unitree_legged_msgs.msg import LowCmd
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import MotorCmd
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import time
from std_msgs.msg import Float32MultiArray

import onnx
import onnxruntime as ort

import math
import time
#### EVERYTHING IS ASSUMED TO BE OMNI ORDER ####

#### SDK ORDER #####
# 0 FR_hip_joint   #
# 1 FR_thigh_joint #
# 2 FR_calf_joint  #
# 3 FL_hip_joint   #
# 4 FL_thigh_joint #
# 5 FL_calf_joint  #
# 6 RR_hip_joint   #
# 7 RR_thigh_joint #
# 8 RR_calf_joint  #
# 9 RL_hip_joint   #
# 0 RL_thigh_joint #
# 1 RL_calf_joint  #
####################

### OMNI ORDER #####
# 0 FL_hip_joint   #
# 1 FR_hip_joint   #
# 2 RL_hip_joint   #
# 3 RR_hip_joint   #
# 4 FL_thigh_joint #
# 5 FR_thigh_joint #
# 6 RL_thigh_joint #
# 7 RR_thigh_joint #
# 8 FL_calf_joint  #
# 9 FR_calf_joint  #
# 0 RL_calf_joint  #
# 1 RR_calf_joint  #
####################

joint_limit_tolerance = 0.05

Hip_max = 1.047 - joint_limit_tolerance
Hip_min = -1.047 + joint_limit_tolerance
Thigh_max = 2.966 - joint_limit_tolerance
Thigh_min = -0.663 + joint_limit_tolerance
Calf_max = -0.837 - joint_limit_tolerance
Calf_min = -2.721 + joint_limit_tolerance

min_limits = [Hip_min, Thigh_min, Calf_min, Hip_min, Thigh_min, Calf_min, Hip_min, Thigh_min, Calf_min, Hip_min, Thigh_min, Calf_min] # SDK ORDER
max_limits = [Hip_max, Thigh_max, Calf_max, Hip_max, Thigh_max, Calf_max, Hip_max, Thigh_max, Calf_max, Hip_max, Thigh_max, Calf_max] # SDK ORDER

SDK_STRINGS = ["FR_hip_joint","FR_thigh_joint","FR_calf_joint","FL_hip_joint","FL_thigh_joint","FL_calf_joint","RR_hip_joint","RR_thigh_joint","RR_calf_joint","RL_hip_joint","RL_thigh_joint","RL_calf_joint"]
OMNI_STRINGS = ["FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint","FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint","FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint"]

omni_to_sdk = []
for i in SDK_STRINGS:
    omni_to_sdk.append(OMNI_STRINGS.index(i))

sdk_to_omni = []
for i in OMNI_STRINGS:
    sdk_to_omni.append(SDK_STRINGS.index(i))

device = 'cuda'

gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=device)

onnx_providers = ['CUDAExecutionProvider']
#onnx_providers = ['TensorrtExecutionProvider']

state_lock = threading.Lock()
odom_lock = threading.Lock()
cmd_lock = threading.Lock()
cmd_vel_lock = threading.Lock()

def joint_limit_clamp(in_vals):
    out_vals = []
    for i in range(len(in_vals)):
        out_vals.append(max(min_limits[i], min(max_limits[i], in_vals[i])))

    return out_vals

def do_clip_actions(actions, clamp_low=-1.0, clamp_high=1.0):
    clipped_actions = []

    for a in actions:
        clipped_actions.append(max(clamp_low, min(clamp_high, a)))

    return clipped_actions

def remap_order(j_in, indices):
    if type(j_in) == torch.Tensor:
        j_values = j_in.tolist()
    else:
        j_values = j_in

    j_out = []

    for i in indices:
        j_out.append(j_values[i])

    return j_out

def state_callback(msg):
    global current_joint_positions, current_joint_velocities, state_init, state_lock

    temp_joint_positions = []
    temp_joint_velocities = []

    for i in sdk_to_omni:
        j = msg.motorState[i]
        temp_joint_positions.append(j.q)
        temp_joint_velocities.append(j.dq)

    state_lock.acquire()
    current_joint_positions = temp_joint_positions
    current_joint_velocities = temp_joint_velocities
    state_init = True
    state_lock.release()

def odom_callback(msg):
    global current_odom, odom_init, odom_lock

    odom_lock.acquire()
    current_odom = msg
    odom_init = True
    odom_lock.release()

# Blatantly stolen from omni.isaac.core.utils.torch.rotations
@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

# Blatantly stolen from omni.isaac.core.utils.torch.rotations
@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def get_transform():
    global current_odom, odom_lock

    odom_lock.acquire()
    pos = current_odom.pose.pose.position
    ori = current_odom.pose.pose.orientation
    odom_lock.release()

    return [pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z]

def get_velocity():
    global current_odom, odom_lock

    odom_lock.acquire()
    linear = current_odom.twist.twist.linear
    rpy = current_odom.twist.twist.angular
    odom_lock.release()

    return [linear.x, linear.y, linear.z], [rpy.x, rpy.y, rpy.z]

def get_joint_positions():
    global state_lock, current_joint_positions

    state_lock.acquire()
    temp_joint_positions = current_joint_positions
    state_lock.release()

    return temp_joint_positions

def get_joint_velocities():
    global state_lock, current_joint_velocities

    state_lock.acquire()
    temp_joint_velocities = current_joint_velocities
    state_lock.release()

    return temp_joint_velocities

def get_commands():
    global cmd_vel_lock, current_cmd_vel

    cmd_vel_lock.acquire()
    temp_cmd_vel = current_cmd_vel
    cmd_vel_lock.release()

    return [temp_cmd_vel.linear.x, temp_cmd_vel.angular.z]

def cmd_vel_callback(msg):
    global cmd_vel_lock, current_cmd_vel

    cmd_vel_lock.acquire()
    current_cmd_vel = msg
    cmd_vel_lock.release()

def get_observations(lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions, default_dof_pos, clip_observations):
    commands = torch.tensor(get_commands(), dtype=torch.float, device=device)
    torso_position, torso_rotation = get_transform()

    torso_rotation = torch.tensor([torso_rotation], dtype=torch.float, device=device)

    dof_vel = torch.tensor(get_joint_velocities(), dtype=torch.float, device=device)
    dof_pos = torch.tensor(get_joint_positions(), dtype=torch.float, device=device)

    root_velocity, ang_velocity = get_velocity()
    ang_velocity = torch.tensor(ang_velocity, dtype=torch.float, device=device)
    base_ang_vel = ang_velocity * ang_vel_scale

    projected_gravity = quat_rotate_inverse(torso_rotation, gravity_vec)
    dof_pos = dof_pos - default_dof_pos
    
    commands_scaled = commands * torch.tensor(
        [lin_vel_scale, ang_vel_scale],
        requires_grad=False,
        device=device,
    )

    obs = torch.cat(
        (
            base_ang_vel,
            projected_gravity[0],
            commands_scaled,
            dof_pos * dof_pos_scale,
            dof_vel * dof_vel_scale,
            last_actions,
        ),
        dim=-1,
    )

    obs = torch.unsqueeze(obs, 0)
    obs = torch.clamp(obs, min=-clip_observations, max=clip_observations)

    return obs

def set_gains(kp, kd):
    global cmd_lock, current_targets

    cmd_lock.acquire()

    for i in range(12):
        current_targets.motorCmd[i].Kp = kp
        current_targets.motorCmd[i].Kd = kd

    cmd_lock.release()

def control_loop(te):
    global lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions, default_dof_pos, clip_observations, clip_actions, action_scale, control_dt
    global ort_model
    global obs_pub, act_pub

    if not rospy.is_shutdown():
        with torch.no_grad():
            start = time.time_ns() / (10 ** 9)
            obs = get_observations(lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions, default_dof_pos, clip_observations)

            outputs = ort_model.run(None, {"obs": obs.cpu().numpy()}, )

            actions = outputs[0][0]

            actions = do_clip_actions(actions, -clip_actions, clip_actions)

            obs_msg = Float32MultiArray()
            obs_msg.data = obs[0].cpu().tolist()
            obs_pub.publish(obs_msg)

            act_msg = Float32MultiArray()
            act_msg.data = actions
            act_pub.publish(act_msg)

            set_current_actions(actions, action_scale, default_dof_pos)

            last_actions = torch.tensor(actions, dtype=torch.float, device=device)

            end = time.time_ns() / (10 ** 9)

            nn_time = end - start
            if nn_time >= control_dt:
                rospy.logwarn("LOOP EXECUTION TIME (" + str(nn_time) + ")")

def publish_cmd(te):
    global cmd_lock, current_targets, cmd_pub

    if not rospy.is_shutdown():
        cmd_lock.acquire()
        cmd_pub.publish(current_targets)
        cmd_lock.release()

def set_current_targets(targets):
    global cmd_lock, current_targets

    targets_remapped = remap_order(targets, omni_to_sdk)
    targets_clamped = joint_limit_clamp(targets_remapped)

    if not rospy.is_shutdown():
        cmd_lock.acquire()
        for i in range(12):
            current_targets.motorCmd[i].q = targets_clamped[i]

        cmd_lock.release()

def set_current_actions(actions, scale, zeros):
    global cmd_lock, current_targets

    new_targets = []

    if not rospy.is_shutdown():
        for i in omni_to_sdk:
            new_targets.append(zeros[i] + actions[i] * scale)

        new_targets = joint_limit_clamp(new_targets)

        cmd_lock.acquire()

        for i in range(12):
            current_targets.motorCmd[i].q = new_targets[i]

        cmd_lock.release()

def main():
    global state_init, odom_init, cmd_lock, cmd_pub, current_cmd_vel, current_targets
    global lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions, default_dof_pos, clip_observations, clip_actions, action_scale, control_dt
    global ort_model
    global obs_pub, act_pub

    rospy.init_node('go1_omni_controller', anonymous=True)

    state_init = False
    odom_init = False

    lin_vel_scale = rospy.get_param("/env/learn/linearVelocityScale")
    ang_vel_scale = rospy.get_param("/env/learn/angularVelocityScale")
    dof_pos_scale = rospy.get_param("/env/learn/dofPositionScale")
    dof_vel_scale = rospy.get_param("/env/learn/dofVelocityScale")
    named_default_joint_angles = rospy.get_param("/env/defaultJointAngles")
    dof_names = OMNI_STRINGS

    model_path = rospy.get_param("/model/path")

    Kp = rospy.get_param("/env/control/stiffness")
    Kd = rospy.get_param("/env/control/damping")
    action_scale = rospy.get_param("/env/control/actionScale")

    clip_observations = rospy.get_param("/env/clipObservations")
    clip_actions = rospy.get_param("/env/clipActions")

    SDK_RATE = 500.0
    SDK_DT = 1.0/SDK_RATE

    control_freq_inv = rospy.get_param("/env/controlFrequencyInv")
    sim_dt = rospy.get_param("/sim/dt")
    control_rate = 1.0/(sim_dt * control_freq_inv)
    control_dt = 1.0/control_rate

    default_dof_pos = []

    for i in range(len(dof_names)):
        name = dof_names[i]
        angle = named_default_joint_angles[name]
        default_dof_pos.append(angle)

    default_dof_pos = torch.tensor(default_dof_pos, dtype=torch.float, device=device)

    current_targets = LowCmd()
    current_targets.levelFlag = 255 # LOW LEVEL

    for i in range(12):
        current_targets.motorCmd[i].mode = 10

    set_gains(0.0, 0.0)

    set_current_targets(default_dof_pos)
    last_actions = torch.tensor([0.0]*12, dtype=torch.float, device=device)

    rospy.loginfo("ONNX: " + str(onnx.__version__))

    rospy.loginfo("#####################################################")
    rospy.loginfo(model_path)
    rospy.loginfo("#####################################################")

    rospy.loginfo("Verifying Model...")
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    rospy.loginfo("#####################################################")
    rospy.loginfo("Loading Model...")

    rospy.loginfo("Available: " + str(ort.get_available_providers()))

    ort_model = ort.InferenceSession(model_path, providers=onnx_providers)
    rospy.loginfo("Loaded")

    rospy.loginfo("Used: " + str(ort_model.get_providers()))

    rospy.loginfo("#####################################################")

    current_cmd_vel = Twist()

    cmd_pub = rospy.Publisher('/low_cmd', LowCmd, queue_size=10)
    obs_pub = rospy.Publisher('/biped/observations', Float32MultiArray, queue_size=10)
    act_pub = rospy.Publisher('/biped/actions', Float32MultiArray, queue_size=10)
    rospy.Subscriber("/low_state", LowState, state_callback)
    rospy.Subscriber("/odometry/filtered", Odometry, odom_callback)
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)

    rospy.Timer(rospy.Duration(SDK_DT), publish_cmd)

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        rospy.loginfo("Waiting for initialization data...")
        if state_init and odom_init:
            break
        rate.sleep()

    rospy.loginfo("Initialized")

    rate = rospy.Rate(5)
    Kp_step = Kp / 25.0 # 5 Seconds
    Kd_step = Kd / 25.0 # 5 Seconds
    Kd_temp = 0
    Kp_temp = 0

    rospy.loginfo("Standing... (Ramping PD Values)")

    while Kp_temp < Kp:
        set_gains(Kp_temp, Kd_temp)
        rate.sleep()
        Kp_temp = Kp_temp + Kp_step
        Kd_temp = Kd_temp + Kd_step

    set_gains(Kp, Kd)

    rospy.loginfo("Done")

    warmup = True
    rospy.loginfo("Warming Up...")
    warmup_count = 30

    with torch.no_grad():
        while not rospy.is_shutdown():
            start = time.time_ns() / (10 ** 9)
            obs = get_observations(lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions, default_dof_pos, clip_observations)

            outputs = ort_model.run(None, {"obs": obs.cpu().numpy()}, )

            mu = outputs[0][0]

            if warmup:
                warmup_count = warmup_count - 1
                if warmup_count <= 0:
                    warmup = False
                    rospy.loginfo("Go!")
                    break

            end = time.time_ns() / (10 ** 9)

            nn_time = end - start
            rospy.sleep(control_dt - nn_time)

    rospy.loginfo("Starting Control Loop")

    rospy.Timer(rospy.Duration(control_dt), control_loop)
    rospy.spin()

main()


