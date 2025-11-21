from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union
import env.models.AUV_model_fuzzy as model

class BaseMDP (gym.Env, ABC):
    """
    MDP环境基类，提供基础的MDP功能框架
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        # 基础配置
        self.default_config = {
            'max_steps': 1000,
            'stop_steps': 1000,
            'dt': 0.1,
            'random_start': True,
            'use_fuzzy': False,
            'random_seed': None
        }
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # 设置随机种子
        if self.config['random_seed'] is not None:
            np.random.seed(self.config['random_seed'])

        # 动作限制
        self.action_limits = {
            'thrust': (model.AuvModel.THRUST_MIN_AUV, model.AuvModel.THRUST_MAX_AUV),
            'rudder': (-model.AuvModel.RUDDER_MAX_AUV, model.AuvModel.RUDDER_MAX_AUV)
        }

        # 初始化状态变量
        self.x = np.zeros((6, 1))
        self.u = np.zeros((6, 1))
        self.delta = np.zeros((3, 1))
        self.f = np.zeros((6, 1))
        self.t = 0
        self._count = 0

        # 初始化空间
        self._setup_spaces()

    
    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境到初始状态"""
        if seed is not None:
            np.random.seed(seed)
        
        self.t = 0
        self._count = 0
        
        # 重置状态变量
        self._reset_state()
        
        # 如果启用随机训练，随机化初始状态
        if self.config['random_start']:
            self._randomize_initial_state()
            
        # 设置模糊参数
        self._setup_fuzzy_parameters()
        
        obs = self._get_obs()
        info = {
            'initial_state': self.x.copy(),
            'time': self.t,
            'step': self._count
        }
        
        return obs, info
    
    def _reset_state(self):
        """重置状态变量"""
        self.x = np.zeros((6, 1))
        self.u = np.zeros((6, 1))
        self.delta = np.zeros((3, 1))
        self.f = np.zeros((6, 1))
        
    @abstractmethod
    def _randomize_initial_state(self):
        """随机化初始状态"""
        pass
    
    @abstractmethod
    def _setup_spaces(self):
        """设置动作和观察空间"""
        pass

    @abstractmethod
    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """计算奖励和相关信息"""
        pass

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """获取观察值"""
        pass

    def runge_kutta(self) -> Tuple[np.ndarray, np.ndarray]:
        """四阶龙格库塔法求解状态方程"""
        auv = model.AuvModel(
            self.x, self.u, self.delta, self.f,
            self.t, self.random_factor_array,
            self.config['use_fuzzy']
        )
        
        dt = self.config['dt']
        k1 = auv.get_du(self.x, self.u, self.delta, self.f)
        k2 = auv.get_du(self.x, self.u + 0.5*dt*k1, self.delta, self.f)
        k3 = auv.get_du(self.x, self.u + 0.5*dt*k2, self.delta, self.f)
        k4 = auv.get_du(self.x, self.u + dt*k3, self.delta, self.f)
        
        # 更新速度
        self.u = self.u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        # 更新位置和姿态
        trans = model.CoordinateTrans(self.x, self.u)
        dx = trans.mov_to_fix(self.x, self.u)
        x = dx * dt + self.x
        
        return x, self.u

    @staticmethod
    def _limit_angle(angle: float) -> float:
        """将角度限制在[-pi, pi]范围内"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _setup_fuzzy_parameters(self):
        """设置模糊参数"""
        if self.config['use_fuzzy']:
            self.random_factor_array = np.random.uniform(0.8, 1.2, 25)
        else:
            self.random_factor_array = np.ones(25)

    @abstractmethod
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步环境交互
        
        Returns:
            Tuple containing:
            - observation (np.ndarray): 环境观察值
            - reward (float): 奖励值
            - terminated (bool): 是否因完成任务而终止
            - truncated (bool): 是否因达到步数限制而截断
            - info (dict): 额外信息
        """
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
        
        # 计算奖励
        reward, reward_info = self._compute_reward()
        
        # 更新计数器
        self.t += self.config['dt']
        self._count += 1
        
        # 检查终止状态
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        # 构建信息字典
        info = {
            'reward_info': reward_info,
            'state': {
                'position': self.x[:3].flatten(),
                'orientation': self.x[3:].flatten(),
                'velocity': self.u.flatten(),
            },
            'time': self.t,
            'step': self._count
        }
        
    
        
        return self._get_obs(), reward, terminated, truncated, info

    @abstractmethod
    def _check_terminated(self) -> bool:
        """检查是否因完成任务而终止"""
        return False

    @abstractmethod
    def _check_truncated(self) -> bool:
        """检查是否因达到步数限制而截断"""
        return self._count >= self.config['stop_steps']
    
