import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

class RandomTraj3D:
    def __init__(
        self,
        traj_type: str = 'random',  # 轨迹类型 ['random', 'inner_helix', 'outer_helix', 'polyline', 'spline']
        # 公共参数
        max_step: int = 1002,
        # 螺旋轨迹参数
        helix_config: dict = {
            'base_radius': 30.0,      # 基础半径
            'radius_change_rate': 0.2, # 半径变化率 (对应0.2)
            'z_change_rate': 0.3,     # Z轴变化率 (对应0.3)
        },
        # 折线轨迹参数
        polyline_config: dict = {
            'segment_length': 50,     # 折线分段长度
            'max_angle': np.pi/3,     # 最大转向角
        }
    ):
        self.traj_type = traj_type
        self.max_step = max_step
        self.helix_config = helix_config
        self.polyline_config = polyline_config
        
    def rand_traj(self):
        """主入口函数，根据类型生成轨迹"""
        if self.traj_type == 'inner_helix':
            return self._generate_spiral(inner=True)
        elif self.traj_type == 'outer_helix':
            return self._generate_spiral(inner=False)
        elif self.traj_type == 'polyline':
            return self._generate_polyline()
        else:
            return self._generate_random()

    def _generate_spiral(self, inner: bool=True):
        """根据用户指定公式生成螺旋轨迹"""
        # 参数初始化（直接根据max_step设置时间参数）
        t = np.linspace(0, 100.2, self.max_step)  # 直接生成max_step个点
        
        # 随机系数
        z_coef = np.random.uniform(1, 2)
        
        # 半径计算（线性变化）
        base_radius = self.helix_config.get('base_radius', 30)
        radius_change_per_step = self.helix_config['radius_change_rate'] * 100.2 / self.max_step
        if inner:
            radius = base_radius - radius_change_per_step * np.arange(self.max_step)
        else:
            radius = base_radius + radius_change_per_step * np.arange(self.max_step)
        
        # 角度参数（保持总旋转角度与原始一致）
        angle_rate = np.pi / 25 * 100.2 / (self.max_step/10)  # 调整角度变化率
        
        # 生成坐标
        x = radius * np.sin(angle_rate * t)
        y = radius * np.cos(angle_rate * t)
        z = 60 - self.helix_config['z_change_rate'] * t * z_coef
        
        return x.tolist(), y.tolist(), z.tolist()

    def _generate_polyline(self):
        """三维折线轨迹"""
        x, y, z = [0], [0], [0]
        current_dir = np.random.randn(3)
        current_dir /= np.linalg.norm(current_dir)  # 初始方向
        
        # 控制分段数量在5-8段之间
        num_segments = np.random.randint(5, 9)
        segment_length = self.max_step // num_segments  # 计算每段理论长度
        remaining_steps = self.max_step % num_segments  # 处理余数
        
        for seg in range(num_segments):
            # 生成新方向
            delta_angle = np.random.uniform(-self.polyline_config['max_angle'], 
                                           self.polyline_config['max_angle'], 3)
            rotation = Rotation.from_euler('xyz', delta_angle)
            current_dir = rotation.apply(current_dir)
            
            # 最后一段处理余数
            actual_length = segment_length + (1 if seg < remaining_steps else 0)
            
            # 生成固定长度的线段
            t = np.linspace(0, actual_length, actual_length)
            new_x = x[-1] + current_dir[0] * t
            new_y = y[-1] + current_dir[1] * t 
            new_z = z[-1] + current_dir[2] * t
            
            x.extend(new_x.tolist())
            y.extend(new_y.tolist())
            z.extend(new_z.tolist())
            
        return x[:self.max_step], y[:self.max_step], z[:self.max_step]

    def _generate_random(self):
        """原始随机轨迹生成方法"""
        # 初始参考轨迹位置
        x_desire = [0]  # 初始x坐标
        y_desire = [0]  # 初始y坐标
        z_desire = [0]

        # 初始航向角和距离
        theta = np.random.uniform(-np.pi / 3, np.pi / 3)
        phi = np.random.uniform(-np.pi / 3, np.pi / 3)
        d = np.random.uniform(0.2, 0.3)

        # 生成随机参考点
        for i in range(self.max_step):
            # 航向角和前进距离的平滑随机扰动
            if i % 50 == 0:
                delta_theta = np.random.uniform(-np.pi/120, np.pi/120)
                delta_phi = np.random.uniform(-np.pi/180, np.pi/180)
                delta_d = np.random.uniform(-0.05, 0.05)
            
            theta = np.clip(theta + delta_theta, -np.pi / 3, np.pi / 3)
            phi = np.clip(phi + delta_phi, -np.pi / 3, np.pi / 3)
            d = np.clip(d + delta_d, 0.2, 0.3)

            # 计算下一个参考点
            x_new = x_desire[-1] + np.cos(theta) * d
            y_new = y_desire[-1] + np.sin(theta) * d
            z_new = z_desire[-1] + np.sin(phi) * d
            
            x_desire.append(x_new)
            y_desire.append(y_new)
            z_desire.append(z_new)

        return x_desire, y_desire, z_desire
    
if __name__ == "__main__":
    # 创建测试用的轨迹类型配置
    traj_types = [
        ('random', '随机轨迹', 'gray'),
        ('inner_helix', '内螺旋', 'red'),
        ('outer_helix', '外螺旋', 'blue'),
        ('polyline', '三维折线', 'green'),
    ]
    
    # 创建3D绘图画布
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle("3D Trajectory Type Comparison", fontsize=16)
    
    # 生成并绘制每种轨迹
    for idx, (traj_type, title, color) in enumerate(traj_types, 1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        
        # 生成轨迹实例（修正helix_config配置）
        generator = RandomTraj3D(
            traj_type=traj_type,
            helix_config={
                'base_radius': 40,
                'radius_change_rate': 0.3,
                'z_change_rate': 0.25
            },
            polyline_config={'segment_length': 30, 'max_angle': np.pi/4},
            max_step=1002
        )
        x, y, z = generator.rand_traj()
        
        # 绘制轨迹
        ax.plot(x, y, z, color=color, linewidth=1.5, alpha=0.8)
        ax.set_title(f"{title} ({traj_type})", fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        ax.set_zlabel('Depth (m)')
        ax.invert_zaxis()
        # ax.set_zlim(60, 0)
        ax.view_init(elev=20, azim=-45)  # 设置视角
        
    plt.tight_layout()
    plt.show()
