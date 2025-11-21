"""
AUV轨迹跟踪与障碍物避障环境
集成了障碍物避障、洋流干扰和轨迹跟踪功能
"""
from env.core.mdp_base import BaseMDP
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from env.utils.AvoidObstacles import AvoidObstacles
from env.utils.ocean_current import OceanCurrent
from env.renderers.auv_renderer import AUVRenderer


class AUVTrackingObstacleAvoidanceEnv(BaseMDP):
    """
    AUV轨迹跟踪与障碍物避障任务环境
    
    该环境在轨迹跟踪的基础上增加了：
    1. 障碍物生成和避障轨迹规划
    2. 洋流干扰（可选）
    3. 障碍物碰撞检测
    4. 避障奖励机制
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化环境
        
        Args:
            config: 配置字典，可包含以下参数：
                - obstacle_enabled: 是否启用障碍物（默认True）
                - r_obstacle: 障碍物半径（默认3.0）
                - avoid_dis: 避障安全距离（默认3.0）
                - clip_parts: 避障轨迹分段数（默认5）
                - current_enabled: 是否启用洋流（默认False）
                - current_config: 洋流配置参数
                - collision_penalty: 碰撞惩罚（默认-100）
                - safe_distance_reward_weight: 安全距离奖励权重（默认1.0）
        """
        # 设置默认配置
        default_obstacle_config = {
            'obstacle_enabled': True,
            'r_obstacle': 3.0,
            'avoid_dis': 3.0,
            'clip_parts': 5,
            'current_enabled': False,
            'current_config': {
                'mu': 0,
                'Vmin': 0.2,
                'Vmax': 0.6,
                'Vc_init': 0.3,
                'alpha_init': 0.0,
                'beta_init': 0.0,
            },
            'collision_penalty': -100.0,
            'safe_distance_reward_weight': 1.0,
            'reward_weights': {
                'path': 1.0,
                'pitch': 0.5,
                'yaw': 0.5,
                'obstacle': 2.0,
                'collision': 1.0
            }
        }
        
        if config is None:
            config = {}
        
        # 合并配置
        for key, value in default_obstacle_config.items():
            if key not in config:
                config[key] = value
        
        super().__init__(config)
        
        # 初始化期望状态
        self.desired_states = {
            'x': 0, 'y': 0, 'z': 0,
            'yaw': 0, 'pitch': 0
        }
        
        # 初始化路径误差和障碍物信息
        self.d_path = 0
        self.obstacle = None
        self.obstacle_info = None
        self.collision_occurred = False
        
        # 生成参考轨迹（带障碍物）
        self.x_traj = None
        self.y_traj = None
        self.z_traj = None
        self.ref_traj = None
        
        # 洋流
        self.ocean_current = None
        
        # 渲染器
        self.renderer = None
        
        # 前一步的距离（用于计算进步奖励）
        self.prev_d_path = None
        self.prev_obstacle_distance = None

    def _setup_spaces(self):
        """设置动作和观察空间"""
        # 动作空间: [推力, 垂直舵角, 水平舵角]
        self.action_space = spaces.Box(
            low=np.array([
                self.action_limits['thrust'][0],
                self.action_limits['rudder'][0],
                self.action_limits['rudder'][0]
            ]),
            high=np.array([
                self.action_limits['thrust'][1],
                self.action_limits['rudder'][1],
                self.action_limits['rudder'][1]
            ]),
            dtype=np.float32
        )
        
        # 观察空间: 22维状态空间（扩展以包含障碍物信息）
        # [0-2]: 位置误差 (dx, dy, dz)
        # [3]: 路径误差 d_path
        # [4-8]: 速度 (u, v, w, q, r)
        # [9-12]: 期望俯仰角和误差的三角函数
        # [13-16]: 期望偏航角和误差的三角函数
        # [17-19]: 障碍物相对位置 (相对于AUV)
        # [20]: 到障碍物的距离
        # [21]: 碰撞标志
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32
        )

    def _randomize_initial_state(self):
        """随机化初始状态"""
        # 位置随机化
        for i in [0, 1]:
            offset = np.random.choice([-1, 1]) * np.random.rand()
            self.x[i] = float(self.x[i] + offset)
        
        # 艏向角随机化
        offset = np.random.choice([-1, 1]) * (45/180*np.pi) * np.random.rand()
        self.x[-1] = float(self.x[-1] + offset)
        
        # 速度随机化
        self.u[0] = float(1 + np.random.choice([-1, 1]) * np.random.rand()*0.5)
        for i in [1, -1]:
            offset = np.random.choice([-1, 1]) * np.random.rand()*0.1
            self.u[i] = float(self.u[i] + offset)
        
        # 推力和舵角随机化
        self.f[0] = float(np.random.rand()*500)
        offset = np.random.choice([-1, 1]) * np.random.rand() * (25*np.pi/180)
        self.delta[0] = float(self.delta[0] + offset)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 重置基本状态
        self.t = 0
        self._count = 0
        self.collision_occurred = False
        self.prev_d_path = None
        self.prev_obstacle_distance = None
        
        # 重置状态变量
        self._reset_state()
        
        # 设置初始状态 [x, y, z, phi, theta, psi]
        self.x = np.array([-1.0, 20.0, 60.0, 0.0, 0.0, np.pi/4]).reshape(6, 1)
        self.u = np.zeros((6, 1))
        self.delta = np.zeros((3, 1))
        self.f = np.zeros((6, 1))
        
        # 生成带障碍物的参考轨迹
        if self.config['obstacle_enabled']:
            avoid_obstacles = AvoidObstacles(
                r_obstacle=self.config['r_obstacle'],
                avoid_dis=self.config['avoid_dis'],
                clip_parts=self.config['clip_parts']
            )
            
            # 生成避障轨迹
            (self.x_traj, self.y_traj, self.z_traj, 
             self.obstacle, x_inter, y_inter, z_inter, 
             x_last, y_last, z_last, self.obstacle_info) = avoid_obstacles.avoid_obstacles_trajectory()
            
            self.ref_traj = np.array([self.x_traj, self.y_traj, self.z_traj])
        else:
            # 如果不启用障碍物，使用简单的螺旋轨迹
            t = np.arange(0, 100.2, 0.1)
            self.x_traj = (30 - 0.2 * t) * np.sin((np.pi/25) * t)
            self.y_traj = (30 - 0.2 * t) * np.cos((np.pi/25) * t)
            self.z_traj = 60 - 0.6 * t
            self.ref_traj = np.array([self.x_traj, self.y_traj, self.z_traj])
            self.obstacle = None
            self.obstacle_info = None
        
        # 初始化洋流
        if self.config['current_enabled']:
            current_config = self.config['current_config']
            self.ocean_current = OceanCurrent(
                mu=current_config['mu'],
                Vmin=current_config['Vmin'],
                Vmax=current_config['Vmax'],
                Vc_init=np.random.uniform(
                    current_config['Vmin'], 
                    current_config['Vmax']
                ),
                alpha_init=np.random.uniform(-np.pi/4, np.pi/4),
                beta_init=np.random.uniform(-np.pi/4, np.pi/4),
                t_step=self.config['dt']
            )
        else:
            self.ocean_current = None
        
        # 随机初始化状态
        if self.config.get('random_start', True):
            self._randomize_initial_state()
        
        # 设置模糊参数
        self._setup_fuzzy_parameters()
        
        # 获取观察
        obs = self._get_obs()
        
        info = {
            'initial_state': self.x.copy(),
            'reference_trajectory': self.ref_traj.copy(),
            'obstacle_info': self.obstacle_info,
            'obstacle_enabled': self.config['obstacle_enabled'],
            'current_enabled': self.config['current_enabled']
        }
        
        return obs, info

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        计算奖励
        
        奖励组成：
        1. 路径跟踪奖励
        2. 姿态跟踪奖励（俯仰角和偏航角）
        3. 避障奖励（与障碍物的距离）
        4. 碰撞惩罚
        """
        # 1. 计算位置误差
        delta_x = self.x[0] - self.desired_states['x']
        delta_y = self.x[1] - self.desired_states['y']
        delta_z = self.x[2] - self.desired_states['z']
        self.d_path = float((delta_x**2 + delta_y**2 + delta_z**2)**0.5)
        
        # 2. 计算姿态误差
        d_pitch = self._limit_angle(self.desired_states['pitch'] - self.x[4])
        d_yaw = self._limit_angle(self.desired_states['yaw'] - self.x[5])
        
        # 3. 基础奖励：路径跟踪和姿态跟踪
        weights = self.config['reward_weights']
        path_reward = -np.tanh(self.d_path / 10.0)  # 归一化到[-1, 0]
        pitch_reward = np.cos(d_pitch)               # [0, 1]
        yaw_reward = np.cos(d_yaw)                   # [0, 1]
        
        reward = (weights['path'] * path_reward + 
                 weights['pitch'] * pitch_reward + 
                 weights['yaw'] * yaw_reward)
        
        # 4. 障碍物相关奖励和惩罚
        obstacle_reward = 0.0
        collision_penalty = 0.0
        obstacle_distance = float('inf')
        
        if self.config['obstacle_enabled'] and self.obstacle is not None:
            # 计算到障碍物的距离
            current_pos = [self.x[0].item(), self.x[1].item(), self.x[2].item()]
            obstacle_distance = self.obstacle.get_distance(current_pos)
            
            # 检查碰撞
            if self.obstacle.check_collision(current_pos, safety_margin=0.5):
                self.collision_occurred = True
                collision_penalty = self.config['collision_penalty']
            
            # 安全距离奖励（鼓励保持安全距离）
            safe_distance = self.config['avoid_dis']
            if obstacle_distance < safe_distance:
                # 距离障碍物越近，惩罚越大
                obstacle_reward = -weights['obstacle'] * np.exp(-obstacle_distance / safe_distance)
            else:
                # 保持安全距离，给予小额奖励
                obstacle_reward = weights['obstacle'] * 0.1
            
            # 避障进步奖励（鼓励远离障碍物）
            if self.prev_obstacle_distance is not None:
                distance_improvement = obstacle_distance - self.prev_obstacle_distance
                if distance_improvement > 0 and obstacle_distance < safe_distance * 2:
                    obstacle_reward += 0.5 * distance_improvement
            
            self.prev_obstacle_distance = obstacle_distance
        
        reward += obstacle_reward + collision_penalty * weights['collision']
        
        # 5. 路径跟踪进步奖励
        progress_reward = 0.0
        if self.prev_d_path is not None:
            progress = self.prev_d_path - self.d_path
            progress_reward = np.clip(progress * 0.5, -1.0, 1.0)
        self.prev_d_path = self.d_path
        
        reward += progress_reward
        
        # 构建详细的奖励信息
        info = {
            'path_error': self.d_path,
            'pitch_error': float(d_pitch),
            'yaw_error': float(d_yaw),
            'path_reward': float(path_reward),
            'pitch_reward': float(pitch_reward),
            'yaw_reward': float(yaw_reward),
            'obstacle_reward': float(obstacle_reward),
            'collision_penalty': float(collision_penalty),
            'progress_reward': float(progress_reward),
            'obstacle_distance': float(obstacle_distance),
            'collision_occurred': self.collision_occurred,
            'total_reward': float(reward)
        }
        
        return float(reward), info
    
    def _get_obs(self) -> np.ndarray:
        """
        获取观察值
        
        观察维度说明：
        [0-2]: 位置误差 (dx, dy, dz)
        [3]: 路径误差 d_path
        [4-8]: 速度 (u, v, w, q, r)
        [9-12]: 期望俯仰角和误差的三角函数
        [13-16]: 期望偏航角和误差的三角函数
        [17-19]: 障碍物相对位置
        [20]: 到障碍物的距离
        [21]: 碰撞标志
        """
        # 更新期望状态
        self._update_desired_states()
        
        obs = np.zeros(22, dtype=np.float32)
        
        # 位置误差
        obs[0] = float(self.x[0] - self.desired_states['x'])
        obs[1] = float(self.x[1] - self.desired_states['y'])
        obs[2] = float(self.x[2] - self.desired_states['z'])
        
        # 路径误差
        obs[3] = float(self.d_path)
        
        # 速度
        obs[4] = float(self.u[0])
        obs[5] = float(self.u[1])
        obs[6] = float(self.u[2])
        obs[7] = float(self.u[4])
        obs[8] = float(self.u[5])
        
        # 期望俯仰角和误差的三角函数
        obs[9] = float(np.cos(self.desired_states['pitch']))
        obs[10] = float(np.sin(self.desired_states['pitch']))
        obs[11] = float(np.cos(self.desired_states['pitch'] - self.x[4]))
        obs[12] = float(np.sin(self.desired_states['pitch'] - self.x[4]))
        
        # 期望偏航角和误差的三角函数
        obs[13] = float(np.cos(self.desired_states['yaw']))
        obs[14] = float(np.sin(self.desired_states['yaw']))
        obs[15] = float(np.cos(self.desired_states['yaw'] - self.x[5]))
        obs[16] = float(np.sin(self.desired_states['yaw'] - self.x[5]))
        
        # 障碍物信息
        if self.config['obstacle_enabled'] and self.obstacle is not None:
            current_pos = np.array([
                float(self.x[0]),
                float(self.x[1]),
                float(self.x[2])
            ])
            obstacle_pos = self.obstacle.position
            
            # 障碍物相对位置
            relative_pos = obstacle_pos - current_pos
            obs[17:20] = relative_pos.astype(np.float32)
            
            # 到障碍物表面的距离
            obs[20] = float(self.obstacle.get_distance(current_pos))
            
            # 碰撞标志
            obs[21] = float(self.collision_occurred)
        else:
            # 无障碍物时，这些值设为默认
            obs[17:20] = [1000.0, 1000.0, 1000.0]  # 远离的障碍物
            obs[20] = 1000.0  # 很大的距离
            obs[21] = 0.0  # 无碰撞
        
        return obs

    def _update_desired_states(self):
        """更新期望状态（与原版相同的LOS制导算法）"""
        # 更新期望位置
        self.desired_states['x'] = float(self.x_traj[self._count])
        self.desired_states['y'] = float(self.y_traj[self._count])
        self.desired_states['z'] = float(self.z_traj[self._count])
        
        # 计算路径切角
        x_d = self.x_traj[self._count+1] - self.x_traj[self._count]
        y_d = self.y_traj[self._count+1] - self.y_traj[self._count]
        z_d = self.z_traj[self._count+1] - self.z_traj[self._count]
        
        alpha_k = np.arctan2(y_d, x_d)
        beta_k = np.arctan(-z_d/(np.sqrt(x_d**2 + y_d**2) + 1e-8))
        
        # 计算转换矩阵
        R1 = np.array([
            [np.cos(alpha_k), -np.sin(alpha_k), 0],
            [np.sin(alpha_k), np.cos(alpha_k), 0],
            [0, 0, 1],
        ])
        R2 = np.array([
            [np.cos(beta_k), 0, np.sin(beta_k)],
            [0, 1, 0],
            [-np.sin(beta_k), 0, np.cos(beta_k)],
        ])
        R = np.dot(R1, R2).T
        
        # 计算路径误差
        P = np.array([
            float(self.x[0]) - self.desired_states['x'],
            float(self.x[1]) - self.desired_states['y'],
            float(self.x[2]) - self.desired_states['z']
        ])
        Pe = np.dot(R, P)
        ye = float(Pe[1])
        ze = float(Pe[2])
        
        # LOS制导计算期望角度
        delta1 = 2.38 * 2  # AUV长度的2倍作为前视距离
        delta2 = (ye**2 + delta1**2)**0.5
        
        self.desired_states['yaw'] = float(alpha_k + np.arctan((-ye) / delta1))
        self.desired_states['pitch'] = float(beta_k + np.arctan((ze) / delta2))

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境交互"""
        # 应用动作
        self.f[0] = float(action[0])
        self.delta[0] = float(action[1])
        self.delta[2] = float(action[2])
        
        # 更新状态（使用龙格库塔法）
        self.x, self.u = self.runge_kutta()
        
        # 如果启用洋流，添加洋流干扰
        if self.config['current_enabled'] and self.ocean_current is not None:
            nu_c = self.ocean_current(self.x.flatten())
            # 洋流影响位置
            current_velocity = nu_c[:3]  # [u_c, v_c, w_c]
            
            # 将洋流速度转换到地理坐标系并更新位置
            from env.models.AUV_model_fuzzy import CoordinateTrans
            trans = CoordinateTrans(self.x, self.u)
            dx_current = trans.mov_to_fix(self.x, nu_c.reshape(6, 1))
            self.x[:3] += dx_current[:3] * self.config['dt']
            
            # 更新洋流参数（时变洋流）
            self.ocean_current.update(self.config['dt'])
        
        # 应用约束
        self.x[3] = 0  # 横滚角恒为0
        self.x[4] = self._limit_angle(self.x[4])
        self.x[5] = self._limit_angle(self.x[5])
        self.u[3] = 0  # 横滚角速度恒为0
        
        # 计算奖励和信息
        reward, reward_info = self._compute_reward()
        
        # 更新计数器
        self.t += self.config['dt']
        self._count += 1
        
        # 检查是否结束
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # 构建info字典
        info = {
            # 当前状态
            'state': {
                'position': self.x[:3].flatten(),
                'orientation': self.x[3:].flatten(),
                'velocity': self.u.flatten(),
            },
            # 控制输入
            'control': {
                'thrust': float(self.f[0]),
                'rudder_v': float(self.delta[0]),
                'rudder_h': float(self.delta[2]),
            },
            # 期望状态
            'desired': self.desired_states.copy(),
            # 奖励详情
            'reward_info': reward_info,
            # 障碍物信息
            'obstacle': {
                'enabled': self.config['obstacle_enabled'],
                'distance': reward_info['obstacle_distance'],
                'collision': self.collision_occurred
            } if self.config['obstacle_enabled'] else None,
            # 其他信息
            'time': self.t,
            'step': self._count
        }
        
        # 如果发生碰撞，标记为终止
        if self.collision_occurred:
            terminated = True
            info['termination_reason'] = 'collision'
        
        # 如果是终止状态，添加终止点信息
        if terminated or truncated:
            info['terminal'] = {
                'position': self.x[:3].flatten(),
                'desired_position': np.array([
                    self.desired_states['x'],
                    self.desired_states['y'],
                    self.desired_states['z']
                ]),
                'final_error': self.d_path,
                'total_steps': self._count,
                'collision_occurred': self.collision_occurred
            }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _check_terminated(self) -> bool:
        """检查是否终止（碰撞或完成任务）"""
        # 碰撞终止
        if self.collision_occurred:
            return True
        # 达到最大步数
        if self._count >= self.config['max_steps']:
            return True
        return False

    def _check_truncated(self) -> bool:
        """检查是否截断"""
        # 超出安全范围（深度、速度等）
        z = float(self.x[2])
        if z < 0 or z > 100:  # 深度超出范围
            return True
        
        speed = float(np.sqrt(self.u[0]**2 + self.u[1]**2 + self.u[2]**2))
        if speed > 10.0:  # 速度过大
            return True
        
        return False

    def render(self, **kwargs):
        """渲染环境"""
        if self.renderer is None:
            self.renderer = AUVRenderer(self)
        self.renderer.render(**kwargs)

    def close(self):
        """关闭环境"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

