"""
海洋洋流模拟模块
"""
import numpy as np

class OceanCurrent:
    """
    海洋洋流模拟类，用于生成时变的洋流干扰
    
    洋流速度在NED坐标系下表示，然后转换到AUV本体坐标系
    """
    
    def __init__(self, mu=0, Vmin=0.2, Vmax=0.6, Vc_init=0.3, 
                 alpha_init=0.0, beta_init=0.0, t_step=0.1):
        """
        Args:
            mu: 洋流变化率（默认为0表示恒定洋流）
            Vmin: 最小洋流速度
            Vmax: 最大洋流速度
            Vc_init: 初始洋流速度
            alpha_init: 初始水平角度（NED坐标系）
            beta_init: 初始垂直角度（NED坐标系）
            t_step: 时间步长
        """
        self.mu = mu
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Vc = Vc_init
        self.alpha = alpha_init
        self.beta = beta_init
        self.t_step = t_step
        
    def __call__(self, state):
        """
        计算当前状态下AUV本体坐标系中的洋流速度
        
        Args:
            state: AUV状态向量 [x, y, z, phi, theta, psi]
                   其中 phi=横滚角, theta=俯仰角, psi=偏航角
                   
        Returns:
            nu_c: 本体坐标系下的洋流速度 [u_c, v_c, w_c, 0, 0, 0]
        """
        phi = state[3]
        theta = state[4]
        psi = state[5]
        
        # NED坐标系下的洋流速度
        vel_current_NED = np.array([
            self.Vc * np.cos(self.alpha) * np.cos(self.beta),  # North
            self.Vc * np.sin(self.beta),                        # East
            self.Vc * np.sin(self.alpha) * np.cos(self.beta)   # Down
        ])
        
        # 转换到本体坐标系
        vel_current_BODY = np.transpose(self.Rzyx(phi, theta, psi)).dot(vel_current_NED)
        
        # 返回6自由度速度向量（角速度分量为0）
        nu_c = np.array([*vel_current_BODY, 0, 0, 0])
        
        return nu_c
    
    def Rzyx(self, phi, theta, psi):
        """
        计算ZYX欧拉角旋转矩阵（从NED到本体坐标系）
        
        Args:
            phi: 横滚角
            theta: 俯仰角
            psi: 偏航角
            
        Returns:
            R: 3x3旋转矩阵
        """
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        return np.vstack([
            np.hstack([cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth]),
            np.hstack([spsi*cth, cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi]),
            np.hstack([-sth, cth*sphi, cth*cphi])
        ])
    
    def update(self, dt=None):
        """
        更新洋流参数（用于时变洋流）
        
        Args:
            dt: 时间步长（如果为None则使用初始化时的t_step）
        """
        if dt is None:
            dt = self.t_step
            
        # 这里可以添加时变洋流的更新逻辑
        # 例如：随机游走、周期性变化等
        if self.mu > 0:
            # 简单的随机游走模型
            self.Vc += np.random.normal(0, self.mu * dt)
            self.Vc = np.clip(self.Vc, self.Vmin, self.Vmax)
            
            self.alpha += np.random.normal(0, self.mu * dt)
            self.alpha = np.arctan2(np.sin(self.alpha), np.cos(self.alpha))  # 限制在[-pi, pi]
            
            self.beta += np.random.normal(0, self.mu * dt)
            self.beta = np.arctan2(np.sin(self.beta), np.cos(self.beta))

