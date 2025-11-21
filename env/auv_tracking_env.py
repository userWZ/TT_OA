from env.core.mdp_base import BaseMDP
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Any, Optional
from env.utils.RandomTraj3D import RandomTraj3D
from env.renderers.auv_renderer import AUVRenderer

class AUVTrackingEnv(BaseMDP):
    """
    AUV轨迹跟踪任务环境
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # 初始化期望状态
        self.desired_states = {
            'x': 0, 'y': 0, 'z': 0,
            'yaw': 0, 'pitch': 0
        }
        
        # 初始化路径误差
        self.d_path = 0
        
        # 生成参考轨迹
        self.x_traj = None
        self.y_traj = None
        self.z_traj = None
        self.ref_traj = None
        self.renderer = None

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
        
        # 观察空间: 17维状态空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
        )

    def _randomize_initial_state(self):
        """随机化初始状态"""
        # 位置随机化（添加.item()转换）
        for i in [0, 1]:
            self.x[i] = (self.x[i] + np.random.choice([-1, 1]) * np.random.rand()).item()
        
        # 艏向角随机化
        self.x[-1] = (self.x[-1] + np.random.choice([-1, 1]) * (45/180*np.pi) * np.random.rand()).item()
        
        # 速度随机化
        self.u[0] = (1 + np.random.choice([-1, 1]) * np.random.rand()*0.5).item()
        for i in [1, -1]:
            self.u[i] = (self.u[i] + np.random.choice([-1, 1]) * np.random.rand()*0.1).item()
        
        # 推力和舵角随机化
        self.f[0] = np.random.rand()*500
        self.delta[0] = (self.delta[0] + np.random.choice([-1, 1]) * np.random.rand() * (25*np.pi/180)).item()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)
        
        # 从配置获取轨迹参数
        traj_config = self.config.get("traj_config", {})
        available_types = self.config.get("traj_types", ["random"])
        
        # 创建新的轨迹生成器（每次reset随机选择类型）
        self.trajectory_generator = RandomTraj3D(
            traj_type=np.random.choice(available_types),
            max_step=self.config["max_steps"] + 2,  # 保证轨迹长度足够
            helix_config=traj_config.get("helix", {
                'base_radius': 30.0,
                'radius_change_rate': 0.2,
                'z_change_rate': 0.3
            }),
            polyline_config=traj_config.get("polyline", {
                'segment_length': 50,
                'max_angle': np.pi/3
            })
        )
        
        # 生成新的参考轨迹
        self.x_traj, self.y_traj, self.z_traj = self.trajectory_generator.rand_traj()
        self.ref_traj = np.array([self.x_traj, self.y_traj, self.z_traj])
        
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
        }
        
        return obs, info

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """计算奖励"""
        # 计算位置误差
        delta_x = self.x[0] - self.desired_states['x']
        delta_y = self.x[1] - self.desired_states['y']
        delta_z = self.x[2] - self.desired_states['z']
        self.d_path = (delta_x**2 + delta_y**2 + delta_z**2)**0.5
        
        # 计算姿态误差
        d_pitch = self._limit_angle(self.desired_states['pitch'] - self.x[4])
        d_yaw = self._limit_angle(self.desired_states['yaw'] - self.x[5])
        
        path_reward = -np.clip(np.log2(self.d_path + 1e-20), -1, 1)  # 路径奖励
        pitch_reward = np.cos(d_pitch)                               # 俯仰角奖励
        yaw_reward = np.cos(d_yaw)                                  # 偏航角奖励
        # 计算奖励
        reward = path_reward + pitch_reward + yaw_reward
        info = {
            'path_error': self.d_path,
            'pitch_error': d_pitch,
            'yaw_error': d_yaw,
            'path_reward': path_reward,
            'pitch_reward': pitch_reward,
            'yaw_reward': yaw_reward
        }
        
        return reward, info
    
    def _get_obs(self) -> np.ndarray:
        """获取观察值"""
        # 更新期望状态
        self._update_desired_states()
        
        obs = np.zeros(17)
        
        # 位置误差（确保使用标量值）
        obs[0:3] = [
            self.x[0].item() - self.desired_states['x'],
            self.x[1].item() - self.desired_states['y'],
            self.x[2].item() - self.desired_states['z']
        ]
        
        # 路径误差
        obs[3] = self.d_path.item() if isinstance(self.d_path, np.ndarray) else self.d_path
        
        # 速度（确保使用标量值）
        obs[4:9] = [
            self.u[0].item(),
            self.u[1].item(),
            self.u[2].item(),
            self.u[4].item(),
            self.u[5].item()
        ]
        
        # 期望姿态和姿态误差的三角函数（添加.item()转换）
        obs[9] = np.cos(self.desired_states['pitch']).item()
        obs[10] = np.sin(self.desired_states['pitch']).item()
        obs[11] = np.cos(self.desired_states['pitch'] - self.x[4].item()).item()
        obs[12] = np.sin(self.desired_states['pitch'] - self.x[4].item()).item()
        obs[13] = np.cos(self.desired_states['yaw']).item()
        obs[14] = np.sin(self.desired_states['yaw']).item()
        obs[15] = np.cos(self.desired_states['yaw'] - self.x[5].item()).item()
        obs[16] = np.sin(self.desired_states['yaw'] - self.x[5].item()).item()
        
        return obs

    def _update_desired_states(self):
        """更新期望状态"""
        # 更新期望位置
        self.desired_states['x'] = self.x_traj[self._count]
        self.desired_states['y'] = self.y_traj[self._count]
        self.desired_states['z'] = self.z_traj[self._count]
        
        # 计算期望姿态
        x_d = self.x_traj[self._count+1] - self.x_traj[self._count]
        y_d = self.y_traj[self._count+1] - self.y_traj[self._count]
        z_d = self.z_traj[self._count+1] - self.z_traj[self._count]
        
        # 计算路径切角
        alpha_k = np.arctan2(y_d, x_d)
        beta_k = np.arctan(-z_d/(np.sqrt(x_d**2 + y_d**2)))
        
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
            self.x[0] - self.desired_states['x'],
            self.x[1] - self.desired_states['y'],
            self.x[2] - self.desired_states['z']
        ])
        Pe = np.dot(R, P)
        ye = Pe[1].item()
        ze = Pe[2].item()
        
        # 计算期望角度
        delta1 = 2.38 * 2
        delta2 = (ye**2 + delta1**2)**0.5
        self.desired_states['yaw'] = alpha_k + np.arctan((-ye) / delta1)
        self.desired_states['pitch'] = beta_k + np.arctan((ze) / delta2) 

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境交互"""
        # 应用动作
        self.f[0] = action[0]
        self.delta[0] = action[1]
        self.delta[2] = action[2]
        
        # 更新状态
        self.x, self.u = self.runge_kutta()
        
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
                'position': self.x[:3].flatten(),  # [x, y, z]
                'orientation': self.x[3:].flatten(),  # [phi, theta, psi]
                'velocity': self.u.flatten(),  # [u, v, w, p, q, r]
            },
            # 控制输入
            'control': {
                'thrust': float(self.f[0]),
                'rudder_v': float(self.delta[0]),  # 垂直舵角
                'rudder_h': float(self.delta[2]),  # 水平舵角
            },
            # 期望状态
            'desired': {
                'x': self.desired_states['x'],
                'y': self.desired_states['y'],
                'z': self.desired_states['z'],
                'pitch': self.desired_states['pitch'],
                'yaw': self.desired_states['yaw']
            },
            # 误差信息
            'errors': reward_info,
            # 其他信息
            'time': self.t,
            'step': self._count
        }
        
        # 如果是终止状态，添加终止点信息
        if terminated or truncated:
            info['terminal'] = {
                'position': self.x[:3].flatten(),  # 终止位置 [x, y, z]
                'desired_position': np.array([  # 期望终止位置
                    self.desired_states['x'],
                    self.desired_states['y'],
                    self.desired_states['z']
                ]),
                'final_error': self.d_path,  # 终止误差
                'total_steps': self._count  # 总步数
            }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _check_terminated(self) -> bool:
        """检查是否终止"""
        return bool(self._count >= self.config['max_steps'])

    def _check_truncated(self) -> bool:
        """检查是否截断"""
        return bool(self._count >= self.config['stop_steps'])

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

