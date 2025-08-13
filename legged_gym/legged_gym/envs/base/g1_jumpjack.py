from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from termcolor import colored
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import wrap_to_pi, yaw_quat, in_poly_2d, batch_rand_int, diff_quat, quaternion_multiply
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from .lpf import ActionFilterButter, ActionFilterExp, ActionFilterButterTorch
from legged_gym.utils.visualization import SingleLine
from .legged_robot import LeggedRobot
from .curiosity import NHashCuriosity

class G1JumpJack(LeggedRobot):
    cfg : LeggedRobotCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless) 
        
    def _init_buffers(self):
        super()._init_buffers()

        self.drifts = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float, requires_grad=False)
        # import ipdb; ipdb.set_trace()
        self.period_contact = torch_rand_float(self.cfg.env.period_contact[0], self.cfg.env.period_contact[1], (self.num_envs,1), self.device,)
        self.time_since_contact = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float, requires_grad=False)
        self.current_contact_reached = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool, requires_grad=False)
        self.contact_sequence = torch.zeros(self.num_envs, 4, self.cfg.task.num_seq, device=self.device, dtype=torch.long, requires_grad=False) # n_env, n_ee, l_seq
        self.resample_sequence(torch.arange(self.num_envs))
        self.current_contact_goal = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.long, requires_grad=False) # store indices
        # currently, the current_contact_goal and current_contact_reached are synced for all ee
        
        self.curiosity_handler = NHashCuriosity(self.cfg.rewards, self.device)

    def _create_envs(self):
        super()._create_envs()
        self.hand_indices[0] = self._body_list.index("right_wrist_roll_rubber_hand")
        self.hand_indices[1] = self._body_list.index("left_wrist_roll_rubber_hand")

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """   
        super()._draw_debug_vis()
        box_quat = yaw_quat(self.base_quat)
        phase_foot = self.contact_sequence[torch.arange(self.num_envs), :2, self.current_contact_goal[:,0]].float()
         # if 1, put foot on the other side; if 0, original side; cannot have double 1
        for env in range(self.num_envs):
            box_pose = gymapi.Transform(gymapi.Vec3(self.base_pos[env, 0],self.base_pos[env, 1],0), 
                                        r=gymapi.Quat(box_quat[env,0], box_quat[env,1], box_quat[env,2], box_quat[env,3]))
            origin_pose = gymapi.Transform(gymapi.Vec3(self.env_origins[env, 0],self.env_origins[env, 1],0))
            left_goal_y = (phase_foot[env,0] < 0.5) * (self.cfg.task.y_side) + \
                            (phase_foot[env,0] > 0.5) * (-self.cfg.task.y_side) + self.cfg.task.y_stance/2
            right_goal_y = (phase_foot[env,1] < 0.5) * ( -self.cfg.task.y_side) + \
                            (phase_foot[env,1] > 0.5) * (self.cfg.task.y_side) - self.cfg.task.y_stance/2
            box = gymutil.WireframeBBoxGeometry(bbox =torch.Tensor([[-self.cfg.task.x_range,left_goal_y-self.cfg.task.y_tol,0.01],
                                                        [self.cfg.task.x_range,left_goal_y+self.cfg.task.y_tol,0.04]]).to(self.device),
                                                    pose=None, color=(0.2,0.2,0.2))
            gymutil.draw_lines(box, self.gym, self.viewer, self.envs[env], origin_pose)
            box = gymutil.WireframeBBoxGeometry(bbox =torch.Tensor([[-self.cfg.task.x_range,right_goal_y-self.cfg.task.y_tol,0.01],
                                                        [self.cfg.task.x_range,right_goal_y+self.cfg.task.y_tol,0.04]]).to(self.device),
                                                    pose=None, color=(0.2,0.2,0.2))
            gymutil.draw_lines(box, self.gym, self.viewer, self.envs[env], origin_pose)
            if self.contact_sequence[env, 2, self.current_contact_goal[env,0]]:
                box = gymutil.WireframeBBoxGeometry(bbox =torch.Tensor([[-0.2,-0.7,self.base_pos[env,2]+0.],
                                                                        [0.5,-0.3,self.base_pos[env,2]+0.7]]).to(self.device),
                                                    pose=None, color=(0.2,0.2,0.2))
            else:
                box = gymutil.WireframeBBoxGeometry(bbox =torch.Tensor([[-0.2,0.4,self.base_pos[env,2]+0.7],
                                                                        [0.5,0.8,self.base_pos[env,2]+0.]]).to(self.device),
                                                    pose=None, color=(0.2,0.2,0.2)) 
            gymutil.draw_lines(box, self.gym, self.viewer, self.envs[env], box_pose)
            if self.contact_sequence[env, 3, self.current_contact_goal[env,0]]:
                box = gymutil.WireframeBBoxGeometry(bbox =torch.Tensor([[-0.2,0.3,self.base_pos[env,2]+0.],
                                                                        [0.5,0.7,self.base_pos[env,2]+0.7]]).to(self.device),
                                                    pose=None, color=(0.2,0.2,0.2))
            else:
                box = gymutil.WireframeBBoxGeometry(bbox =torch.Tensor([[-0.2,-0.4,self.base_pos[env,2]+0.7],
                                                                        [0.5,-0.8,self.base_pos[env,2]+0.]]).to(self.device),
                                                    pose=None, color=(0.2,0.2,0.2)) 
            gymutil.draw_lines(box, self.gym, self.viewer, self.envs[env], box_pose)
                
            
    @property 
    def next_contact_goal(self):
        return torch.clip(self.current_contact_goal + 1, min=0, max=self.contact_sequence.shape[-1]-1)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.resample_sequence(env_ids)
        self.current_contact_goal[env_ids, :] *= 0
        self.time_since_contact[env_ids, :] *= 0
        self.current_contact_reached[env_ids, :] = False
        self.period_contact = torch_rand_float(self.cfg.env.period_contact[0], self.cfg.env.period_contact[1], (self.num_envs,1), self.device,)
    
    def resample_sequence(self, env_ids):
        if len(env_ids) <= 0:
            return
        self.contact_sequence[env_ids] = torch.randint_like(self.contact_sequence[env_ids], low=0, high=2)
        # YMCA needs sync foot hand and fixed transition: 00/(01or10or00)/00/...
        self.contact_sequence[env_ids,0:2,1::2] = 0
        self.contact_sequence[env_ids,1,0::2] &= (~self.contact_sequence[env_ids,0,0::2]) # no 11
        self.contact_sequence[env_ids,2:] = self.contact_sequence[env_ids,:2].clone() # sync hand

    def post_physics_step(self):
        super().post_physics_step()

        left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos = self.get_ee_pos()
        left_foot_pos_xy = left_foot_pos[:, :2] - self.env_origins[:,:2]
        right_foot_pos_xy = right_foot_pos[:, :2] - self.env_origins[:,:2]
        left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b = self.get_ee_pos_b()

        self.time_since_contact += self.dt * self.current_contact_reached
        # update goals for fulfillment
        _switch = self.time_since_contact > self.period_contact
        self.time_since_contact[_switch] = 0
        self.current_contact_reached[_switch] = 0
        self.current_contact_goal = torch.clip(self.current_contact_goal + _switch * 1, min=0, max=self.contact_sequence.shape[-1]-1)

    def vec3d_rot_baseyaw(self, vec):
        # this is an example on how a vec should be rotated w.r.t. base yaw
        return quat_rotate_inverse(yaw_quat(self.base_quat[:]), vec)
        
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure
        """
        noise_vec = torch.zeros_like(self.obs_buf[0,:self.cfg.env.num_noise])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0 : self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[self.num_actions : 2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[2*self.num_actions : 2*self.num_actions+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[2*self.num_actions+3 : 2*self.num_actions+6] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[2*self.num_actions+6 : 2*self.num_actions+9] = noise_scales.gravity * noise_level
        noise_vec[2*self.num_actions+9 : 2*self.num_actions+16] = 0. # phase and xyyaw
        noise_vec[2*self.num_actions+16 : 3*self.num_actions+16] = 0.0  # previous actions
        # hist
        noise_vec[3*self.num_actions+16 : 4*self.num_actions+16] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[4*self.num_actions+16 : 5*self.num_actions+16] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[5*self.num_actions+16 : 6*self.num_actions+16] = 0.0
        noise_vec[6*self.num_actions+16 : 9*self.num_actions+16] = noise_vec[3*self.num_actions+16 : 6*self.num_actions+16].clone()
        
        return noise_vec
        
    def compute_observations(self):
        """ Computes observations
        """
        #todo: deal with 4 ee
        self.phase =  self.contact_sequence[torch.arange(self.num_envs), :, self.current_contact_goal[:,0]].float()
        self.obs_buf = torch.cat((  
                                    (self.dof_pos - self.dof_bias) * self.obs_scales.dof_pos, # 23
                                    self.dof_vel * self.obs_scales.dof_vel, # 23
                                    self.base_ang_vel  * self.obs_scales.ang_vel, # 3
                                    self.base_lin_vel * self.obs_scales.lin_vel, # 3
                                    self.projected_gravity, # 3
                                    self.phase, # 4
                                    self.global_obs(), # 3
                                    self.actions, # 23,
                                    self.joint_hist[:,1,:], # 69
                                    self.joint_hist[:,2,:], # 69
                                    ),dim=-1)
                    
        obs_buf_denoise = self.obs_buf.clone()
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        if self.cfg.env.num_privileged_obs is not None:
            # privileged obs
            self.privileged_info = torch.cat([
                self._base_com_bias,
                self._ground_friction_values[:, self.feet_indices],
                self._link_mass_scale,
                self._kp_scale,
                self._kd_scale,
                self._rfi_lim_scale,
                self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, 6),
            ], dim=1)
            self.privileged_obs_buf = torch.cat([obs_buf_denoise, self.privileged_info], dim=1)
        
        
    def symmetric_dof(self, prev_dof):
        assert prev_dof.shape[1] == 23, print(prev_dof.shape[1])
        
        LEFT_POS_INDICES = [0, 3, 4, 13, 16]    # 左侧直接交换的关节 (pitch类)
        LEFT_NEG_INDICES = [1, 2, 5, 14, 15, 17] # 左侧需取反的关节 (roll/yaw类)
        RIGHT_POS_INDICES = [6, 9, 10, 18, 21]   # 右侧直接交换的关节 (pitch类)
        RIGHT_NEG_INDICES = [7, 8, 11, 19, 20, 22] # 右侧需取反的关节 (roll/yaw类)
        
        new_dof = prev_dof.clone()
        
        new_dof[:, 12] *= -1.  # waist_yaw_joint
        
        # 1. 将右侧POS关节的值复制到左侧POS位置
        new_dof[:, LEFT_POS_INDICES] = prev_dof[:, RIGHT_POS_INDICES].clone()
        # 2. 将右侧NEG关节的值取反后复制到左侧NEG位置
        new_dof[:, LEFT_NEG_INDICES] = prev_dof[:, RIGHT_NEG_INDICES].clone() * (-1.)
        # 3. 将左侧POS关节的值复制到右侧POS位置
        new_dof[:, RIGHT_POS_INDICES] = prev_dof[:, LEFT_POS_INDICES].clone()
        # 4. 将左侧NEG关节的值取反后复制到右侧NEG位置
        new_dof[:, RIGHT_NEG_INDICES] = prev_dof[:, LEFT_NEG_INDICES].clone() * (-1.)
        return new_dof.clone()
        
        
    def symmetric_fn(self, batch_obs, batch_critic_obs, batch_actions):
        assert not self.privileged_obs_buf
        batch_size = batch_obs.shape[0]
        extend_batch_obs = torch.cat([batch_obs, batch_obs], dim=0)
        extend_batch_critic_obs = torch.cat([batch_critic_obs, batch_critic_obs], dim=0)
        extend_batch_actions = torch.cat([batch_actions, batch_actions], dim=0)
        
        # 左-右对称变换动作
        extend_batch_actions[batch_size:, :] = self.symmetric_dof(extend_batch_actions[:batch_size, :])
        # 关节位置 (0-22)
        extend_batch_obs[batch_size:, :23] = self.symmetric_dof(extend_batch_obs[:batch_size, :23])
        # 关节速度 (23-45)
        extend_batch_obs[batch_size:, 23:46] = self.symmetric_dof(extend_batch_obs[:batch_size, 23:46])
        # 基座方向 (假设在46-49)
        extend_batch_obs[batch_size:, 46:47] *= -1.  # 旋转x分量取反
        extend_batch_obs[batch_size:, 48:49] *= -1.  # 旋转z分量取反
        
        # 线速度 (假设在50-52)
        extend_batch_obs[batch_size:, 50:51] *= -1.  # y方向线速度取反
        
        # 重力方向 (假设在53-55)
        extend_batch_obs[batch_size:, 53:54] *= -1.  # y方向重力分量取反
        
        # 相位信息 (假设在56-59)
        # 交换左右腿相位
        extend_batch_obs[batch_size:, 55] = extend_batch_obs[:batch_size, 57].clone()
        extend_batch_obs[batch_size:, 56] = extend_batch_obs[:batch_size, 56].clone()
        extend_batch_obs[batch_size:, 57] = extend_batch_obs[:batch_size, 59].clone()
        extend_batch_obs[batch_size:, 58] = extend_batch_obs[:batch_size, 58].clone()
        
        # 命令信息 (假设在60-62: x速度, y速度, 偏航角速度)
        extend_batch_obs[batch_size:, 60] *= -1.  # y速度取反
        extend_batch_obs[batch_size:, 61] *= -1.  # 偏航角速度取反
        
        # 对称变换历史动作 (假设从63开始)
        extend_batch_obs[batch_size:, 62:85] = self.symmetric_dof(extend_batch_obs[:batch_size, 62:85])
        extend_batch_obs[batch_size:, 85:108] = self.symmetric_dof(extend_batch_obs[:batch_size, 85:108])
        extend_batch_obs[batch_size:, 108:131] = self.symmetric_dof(extend_batch_obs[:batch_size, 108:131])
        extend_batch_obs[batch_size:, 131:154] = self.symmetric_dof(extend_batch_obs[:batch_size, 131:154])
        extend_batch_obs[batch_size:, 154:177] = self.symmetric_dof(extend_batch_obs[:batch_size, 154:177])
        extend_batch_obs[batch_size:, 177:200] = self.symmetric_dof(extend_batch_obs[:batch_size, 177:200])
        extend_batch_obs[batch_size:, 200:223] = self.symmetric_dof(extend_batch_obs[:batch_size, 200:223])

        # 对称变换critic观测值
        extend_batch_critic_obs[batch_size:, :] = extend_batch_obs[batch_size:, :].clone()
        
        return extend_batch_obs, extend_batch_critic_obs, extend_batch_actions
        
    def _whole_body_fulfillment(self):
        base2feet = torch.amax(self.root_states[:, 2:3] - self.foot_height, dim=1)
        height_good = base2feet > 0.65
        
        left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b = self.get_ee_pos_yawb()
        phase =  self.contact_sequence[torch.arange(self.num_envs), :, self.current_contact_goal[:,0]].float()
        lh_ff = torch.logical_or((phase[:,2]<0.5)*(left_hand_pos_b[:,1]>0.4), 
                                    (phase[:,2]>0.5) * (left_hand_pos_b[:,1]<-0.3))
        rh_ff = torch.logical_or((phase[:,3]<0.5)*(right_hand_pos_b[:,1]<-0.4), 
                                    (phase[:,3]>0.5) * (right_hand_pos_b[:,1]>0.3))
        lf_pos_correct, rf_pos_correct = self.correct_foot_xy_ymca()
        lf_ff = lf_pos_correct
        rf_ff = rf_pos_correct
        ff = torch.logical_and(lh_ff, rh_ff) * torch.logical_and(lf_ff, rf_ff)
        wb_good = height_good * ff
        return wb_good
    
    @property
    def knee_distance(self):
        left_knee_pos = self._get_rigid_body_pos("left_knee_link")
        right_knee_pos = self._get_rigid_body_pos("right_knee_link")
        dist_knee = torch.norm(left_knee_pos - right_knee_pos, dim=-1, keepdim=True)
        return dist_knee
    
    @property
    def feet_distance(self):
        left_foot_pos = self._get_rigid_body_pos("left_ankle_roll_link")
        right_foot_pos = self._get_rigid_body_pos("right_ankle_roll_link")
        dist_feet = torch.norm(left_foot_pos - right_foot_pos, dim=-1, keepdim=True)
        return dist_feet

    # contact sequence:  if one: foot another side / hand to another;  if zero: foot original side / arm spread --- towards YMCA
    def _reward_on_box(self):
        """ Reward function to minimize the distance between the robot and the box
        """
        phase =  self.contact_sequence[torch.arange(self.num_envs), :, self.current_contact_goal[:,0]].float()
        contact_left = (self.contact_forces[:, self.feet_indices[0], 2] > 1.)
        contact_right= (self.contact_forces[:, self.feet_indices[1], 2] > 1.)
        contact_lhand = self.contact_forces[:, self.hand_indices[0], :3].norm(dim=-1) > 1.
        contact_rhand = self.contact_forces[:, self.hand_indices[1], :3].norm(dim=-1) > 1.
        left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b = self.get_ee_pos_yawb()
        lf_pos_correct, rf_pos_correct = self.correct_foot_xy_ymca()

        correct_left = contact_left
        correct_left &= lf_pos_correct
        correct_right = contact_right
        correct_right &= rf_pos_correct
        
        correct_hand = torch.ones_like(contact_left).bool() # if there is any one: clap hands
        correct_hand[(phase[:,2] + phase[:,3])>0.5] &= self.hand_touch()[(phase[:,2] + phase[:,3])>0.5]
        correct_hand[(phase[:,2] + phase[:,3])<0.5] &= (~self.hand_touch()[(phase[:,2] + phase[:,3])<0.5])
        
        correct_all = torch.logical_and(correct_left, correct_right)
        correct_all &= correct_hand
        correct_all &= self._whole_body_fulfillment()

        # synced
        self.current_contact_reached[:,0] |= correct_all

        success_feet = correct_left * 1. + correct_right * 1. + correct_hand * 2. + correct_all * 16.0 * 2
        wrong_contact = contact_left * (~correct_left) * 1.0 + contact_right * (~correct_right) * 1.0 + \
                        contact_lhand * (~correct_hand) * 1.0 + contact_rhand * (~correct_hand) * 1.0
        wrong_contact *= (torch.sum(self.current_contact_goal, dim=-1) > 0.5) # penalty only after first time goal reaching

        return (success_feet - wrong_contact * 4.0) * 1.0

    def _reward_prev_fulfill(self):
        return torch.sum(self.current_contact_goal, dim=-1) * 1.0 * self._whole_body_fulfillment()  # previous fulfillment

    def _reward_feet_ori_contact(self):
        contact_left = (self.contact_forces[:, self.feet_indices[0], 2] > 1.)
        contact_right= (self.contact_forces[:, self.feet_indices[1], 2] > 1.)
        left_quat = self._rigid_body_rot[:, self.feet_indices[0]-1]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec) * (contact_left).unsqueeze(-1)
        right_quat = self._rigid_body_rot[:, self.feet_indices[1]-1]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec) * (contact_right).unsqueeze(-1)
        return torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5 

    def _reward_contact_velo(self):
        contact_left  = (self.contact_forces[:, self.feet_indices[0], 2] > 1.)
        contact_right = (self.contact_forces[:, self.feet_indices[1], 2] > 1.)
        foot_vel_left = self._rigid_body_vel[:, self.feet_indices[0]-1, :2]
        foot_vel_right= self._rigid_body_vel[:, self.feet_indices[1]-1, :2]
        _left_error = torch.sum(torch.square(foot_vel_left - 0.), dim=-1) * contact_left
        _right_error = torch.sum(torch.square(foot_vel_right - 0.), dim=-1) * contact_right
        return _left_error + _right_error

    def _reward_rot_z(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_curiosity(self):
        lf_pos, rf_pos, lh_pos, rh_pos = self.get_ee_pos()
        curio_obs = torch.cat([self.base_pos-self.env_origins, self.base_quat, self.root_states[:,7:13].clone(),
                                lf_pos-self.env_origins, rf_pos-self.env_origins, lh_pos-self.env_origins, rh_pos-self.env_origins,
                                self.contact_forces[:,self.feet_indices,:].norm(dim=-1)>0.1,
                                self.hand_touch().unsqueeze(1).repeat(1,2).float()
                                ], dim=-1)
        assert curio_obs.shape[1] == self.cfg.rewards.curiosity.obs_dim
        shift = self.current_contact_goal[:,0:1].clone()
        return self.curiosity_handler.update_curiosity(curio_obs, shift)
    
    def _reward_hand_x(self):
        left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b = self.get_ee_pos_yawb()
        return (left_hand_pos_b[:,0]>0.05)*1.0 + (right_hand_pos_b[:,0]>0.05)*1.0
    
    def _reward_hand_y(self):
        phase =  self.contact_sequence[torch.arange(self.num_envs), :, self.current_contact_goal[:,0]].float()
        left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b = self.get_ee_pos_yawb()
        left_spread = left_hand_pos_b[:,1] * (phase[:,2]<0.5) - left_hand_pos_b[:,1] * (phase[:,2]>0.5)
        right_spread = - right_hand_pos_b[:,1] * (phase[:,3]<0.5) + right_hand_pos_b[:,1] * (phase[:,3]>0.5)
        return left_spread + right_spread
    
    def _reward_hand_z(self):
        left_foot_pos_b, right_foot_pos_b, left_hand_pos_b, right_hand_pos_b = self.get_ee_pos_yawb()
        return torch.abs(left_hand_pos_b[:,2]-0.5) + torch.abs(right_hand_pos_b[:,2]-0.5) 
    
    def _reward_foot_goal(self):
        left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos = self.get_ee_pos()
        left_foot_pos_xy = left_foot_pos[:, :2] - self.env_origins[:,:2]
        right_foot_pos_xy = right_foot_pos[:, :2] - self.env_origins[:,:2]
        phase_foot = self.contact_sequence[torch.arange(self.num_envs), :2, self.current_contact_goal[:,0]].float()
        
        left_goal_y = (phase_foot[:,0] < 0.5) * (self.cfg.task.y_side) + \
                        (phase_foot[:,0] > 0.5) * (-self.cfg.task.y_side) + self.cfg.task.y_stance/2
        right_goal_y = (phase_foot[:,1] < 0.5) * ( -self.cfg.task.y_side) + \
                        (phase_foot[:,1] > 0.5) * (self.cfg.task.y_side) - self.cfg.task.y_stance/2
        left_error = torch.square(left_foot_pos_xy[:,0]/self.cfg.task.x_range) \
                        + torch.square((left_foot_pos_xy[:,1] - left_goal_y)/self.cfg.task.y_tol)
        right_error = torch.square(right_foot_pos_xy[:,0]/self.cfg.task.x_range)\
                        + torch.square((right_foot_pos_xy[:,1] - right_goal_y)/self.cfg.task.y_tol)
        return torch.exp(-left_error-right_error)
        
### utils

    @property
    def foot_height(self):
        foot_height = self._rigid_body_pos[:, self.feet_indices-1, 2]
        return foot_height
    
    def hand_touch(self):
        hand_dist = (self._rigid_body_pos[:, self.hand_indices[0] - 1] - self._rigid_body_pos[:, self.hand_indices[1] - 1]).norm(dim=-1)
        # contact = (self.contact_forces[:, self.hand_indices[0], :2].norm(dim=-1) > 1.) & \
        #                 (self.contact_forces[:, self.hand_indices[1], :2].norm(dim=-1) > 1.)
        # return torch.logical_and(hand_dist < 0.12, contact)
        return hand_dist < 0.10
        
    def global_obs(self):
        # x y yaw
        xy = self.base_pos[:,:2] - self.env_origins[:,:2]
        qx, qy, qz, qw = self.base_quat[:, 0], self.base_quat[:, 1], self.base_quat[:, 2], self.base_quat[:, 3]
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return torch.cat([xy, yaw.unsqueeze(-1)], dim=-1)
    
    def correct_foot_xy_ymca(self):
        left_foot_pos, right_foot_pos, left_hand_pos, right_hand_pos = self.get_ee_pos()
        left_foot_pos_xy = left_foot_pos[:, :2] - self.env_origins[:,:2]
        right_foot_pos_xy = right_foot_pos[:, :2] - self.env_origins[:,:2]
        phase_foot = self.contact_sequence[torch.arange(self.num_envs), :2, self.current_contact_goal[:,0]].float()
        # if 1, put foot on the other side; if 0, original side; cannot have double 1
        
        left_correct_x = torch.abs(left_foot_pos_xy[:,0]) < self.cfg.task.x_range
        left_goal_y = (phase_foot[:,0] < 0.5) * (self.cfg.task.y_side) + \
                        (phase_foot[:,0] > 0.5) * (-self.cfg.task.y_side) + self.cfg.task.y_stance/2
        left_correct_y = torch.abs(left_foot_pos_xy[:,1] - left_goal_y) < self.cfg.task.y_tol

        right_correct_x = torch.abs(right_foot_pos_xy[:,0]) < self.cfg.task.x_range
        right_goal_y = (phase_foot[:,1] < 0.5) * ( -self.cfg.task.y_side) + \
                        (phase_foot[:,1] > 0.5) * (self.cfg.task.y_side) - self.cfg.task.y_stance/2
        right_correct_y = torch.abs(right_foot_pos_xy[:,1] - right_goal_y) < self.cfg.task.y_tol
        
        return left_correct_x*left_correct_y, right_correct_x*right_correct_y