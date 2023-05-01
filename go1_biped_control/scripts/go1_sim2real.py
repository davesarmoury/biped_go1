#!/usr/bin/env python3
import rospy
import torch
from yaml import load
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

device = 'cuda'
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

def get_commands():
    pass
def get_transform():
    pass
def get_velocity():
    pass
def get_joint_positions():
    pass
def get_joint_velocities():
    pass

def odom_callback(msg):
    pass

def cmd_vel_callback(msg):
    pass

def cmd_robot():
    global lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, gravity_vec, default_dof_pos

    commands = get_commands()
    root_transforms = get_transform()
    root_velocity = get_velocity()
    dof_pos = get_joint_positions()
    dof_vel = get_joint_velocities()

    torso_position = root_transforms[0:3]
    torso_rotation = root_transforms[3:6]

    velocity = root_velocity[0:3]
    ang_velocity = root_velocity[3:6]

    base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * ang_vel_scale
    projected_gravity = quat_rotate(torso_rotation, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands * torch.tensor(
        [lin_vel_scale, lin_vel_scale, ang_vel_scale],
        requires_grad=False,
        device=device,
    )

    obs = torch.cat(
        (
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            commands_scaled,
            dof_pos_scaled,
            dof_vel * dof_vel_scale,
            actions,
        ),
        dim=-1,
    )

def main():
    global lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale, gravity_vec, default_dof_pos

    cfg_file = open("/home/davesarmoury/ws/biped_ws/OmniIsaacGymEnvs/omniisaacgymenvs/cfg/task/Go1_Horizontal.yaml", 'r')
    task_cfg = load(cfg_file, Loader=Loader)
    cfg_file.close()

    # normalization
    lin_vel_scale = task_cfg["env"]["learn"]["linearVelocityScale"]
    ang_vel_scale = task_cfg["env"]["learn"]["angularVelocityScale"]
    dof_pos_scale = task_cfg["env"]["learn"]["dofPositionScale"]
    dof_vel_scale = task_cfg["env"]["learn"]["dofVelocityScale"]
    named_default_joint_angles = task_cfg["env"]["defaultJointAngles"]

    default_dof_pos = torch.zeros((len(dof_names)), dtype=torch.float, device=device, requires_grad=False)
    for i in range(len(dof_names)):
        name = dof_names[i]
        angle = named_default_joint_angles[name]
        default_dof_pos[i] = angle

    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=device)

    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("cmd_vel", Twist, cmd_vel_callback)
    rospy.Subscriber("cmd_vel", Odometry, odom_callback)

main()