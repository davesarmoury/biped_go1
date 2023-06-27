#!/usr/bin/env python3
import rospy
import torch
import threading
from unitree_legged_msgs.msg import LowCmd
from unitree_legged_msgs.msg import LowState
from unitree_legged_msgs.msg import MotorCmd
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

#    NETWORK ORDER     SDK INDEX
#
#   "RR_hip_joint"        6
#   "RL_hip_joint"        9
#   "FL_hip_joint"        3
#   "FR_hip_joint"        0
#   "FL_thigh_joint"      4
#   "FR_thigh_joint"      1
#   "RL_thigh_joint"      10
#   "RR_thigh_joint"      7
#   "FL_calf_joint"       5
#   "RL_calf_joint"       11
#   "FR_calf_joint"       2
#   "RR_calf_joint"       8

joint_map = [6, 9, 3, 0, 4, 1, 10, 7, 5, 11, 2, 8]

device = 'cuda'

Kp = 50
Kd = 5

state_lock = threading.Lock()
odom_lock = threading.Lock()
cmd_lock = threading.Lock()

def network_to_sdk(j_in):
    j_out = [None] * 12

    for i in range(12):
        j_out[joint_map[i]] = j_in[i]

    return j_out

def sdk_to_network(j_in):
    j_out = []

    for i in range(12):
        j_out.append(j_in[joint_map[i]])

    return j_out

def state_callback(msg):
    global current_joint_positions, current_joint_velocities, state_init

    temp_joint_positions = []
    temp_joint_velocities = []

    for j in msg.motorState:
        temp_joint_positions.append(j.q)
        temp_joint_velocities.append(j.dq)

    state_lock.acquire()
    current_joint_positions = temp_joint_positions
    current_joint_velocities = temp_joint_velocities
    state_init = True
    state_lock.release()

def odom_callback(msg):
    global current_odom, odom_init

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
    global current_odom, odom_init

    odom_lock.acquire()
    pos = current_odom.pose.pose.position
    ori = current_odom.pose.pose.orientation
    odom_lock.release()

    return [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]

def get_velocity():
    global current_odom, odom_init

    odom_lock.acquire()
    linear = current_odom.twist.twist.linear
    rpy = current_odom.twist.twist.angular
    odom_lock.release()

    return [linear.x, linear.y, linear.z, rpy.x, rpy.y, rpy.z]

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

def get_observations(lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions):
    commands = get_commands()
    root_transforms = get_transform()
    root_velocity = get_velocity()
    dof_pos = sdk_to_network(get_joint_positions())
    dof_vel = sdk_to_network(get_joint_velocities())

    torso_position = root_transforms[0:3]
    torso_rotation = root_transforms[3:7]

    #velocity = root_velocity[0:3]
    ang_velocity = root_velocity[3:6]

    #base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * ang_vel_scale
    projected_gravity = quat_rotate(torso_rotation, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands * torch.tensor(
        [lin_vel_scale, ang_vel_scale],
        requires_grad=False,
        device=device,
    )

    obs = torch.cat(
        (
            #base_lin_vel,
            base_ang_vel,
            projected_gravity,
            commands_scaled,
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            last_actions,
        ),
        dim=-1,
    )

    return obs

def publish_cmd(te):
    global cmd_lock, low_cmd, cmd_pub

    cmd_lock.acquire()
    cmd_pub.publish(low_cmd)
    cmd_lock.release()

def main():
    global state_init, odom_init, low_cmd, cmd_lock, low_cmd, cmd_pub

    rospy.init_node('horizontal_controller', anonymous=True)

    state_init = False
    odom_init = False

    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device)

    lin_vel_scale = rospy.get_param("/env/learn/linearVelocityScale")
    ang_vel_scale = rospy.get_param("/env/learn/angularVelocityScale")
    dof_pos_scale = rospy.get_param("/env/learn/dofPositionScale")
    dof_vel_scale = rospy.get_param("/env/learn/dofVelocityScale")
    named_default_joint_angles = rospy.get_param("/env/defaultJointAngles")
    dof_names = list(named_default_joint_angles.keys())
    model_path = rospy.get_param("/model/path")

    default_dof_pos = []
    default_dof_pos_t = torch.zeros((len(dof_names)), dtype=torch.float, device=device, requires_grad=False)
    for i in range(len(dof_names)):
        name = dof_names[i]
        angle = named_default_joint_angles[name]
        default_dof_pos_t[i] = angle
        default_dof_pos.append(angle)

    last_actions = default_dof_pos
    low_cmd = LowCmd()
    low_cmd.head = [0xEE, 0xEF]
    low_cmd.levelFlag = 0xff  # LOW LEVEL

    for i in range(12):
        low_cmd.motorCmd[i].mode = 0x0A
        low_cmd.motorCmd[i].q = default_dof_pos[joint_map[i]]

        low_cmd.motorCmd[i].Kp = 5.0
        low_cmd.motorCmd[i].Kd = 1.0
        low_cmd.motorCmd[i].tau = 0.0
        low_cmd.motorCmd[i].dq = 0.0
        

    rospy.loginfo("Loading Model...")
    rospy.loginfo(model_path)

    #model = torch.nn.Sequential()
    #checkpoint = torch.load(model_path)
    
    #model.load_state_dict(checkpoint['model'])

    #model.eval()
    #model.to(device)

    rospy.loginfo("Loaded")

    cmd_pub = rospy.Publisher('/low_cmd', LowCmd, queue_size=10)
    rospy.Subscriber("/low_state", LowState, state_callback)
    rospy.Subscriber("/odometry/filtered", Odometry, odom_callback)

    rospy.Timer(rospy.Duration(1.0/500.0), publish_cmd)

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        rospy.loginfo("Waiting for initialization data...")
        if state_init and odom_init:
            break
        rate.sleep()
    
    rospy.loginfo("Initialized")

    rate = rospy.Rate(14)

    with torch.no_grad():
        while not rospy.is_shutdown():
            temp_low_cmd = LowCmd()
            temp_low_cmd.head = [0xFE, 0xEF]
            temp_low_cmd.levelFlag = 0xff  # LOW LEVEL

#            obs = get_observations(lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, last_actions)
#
#            nn_vals = model(obs)
#            last_actions = nn_vals
#            nn_vals_sdk = network_to_sdk(nn_vals)
#
#            for j_q in nn_vals_sdk:
#                tmp_cmd = MotorCmd()
#
 #               tmp_cmd.mode = 0x0A
 #               tmp_cmd.q = j_q
#
#                tmp_cmd.Kp = Kp
#                tmp_cmd.Kd = Kd
#                tmp_cmd.tau = 0.0
#                tmp_cmd.dq = 0.0
#
#                temp_low_cmd.motorCmd.append(tmp_cmd)

#            cmd_lock.acquire()
#            low_cmd = temp_low_cmd
#            cmd_lock.release()

            rate.sleep()

main()