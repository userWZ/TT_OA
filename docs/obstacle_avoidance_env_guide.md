# AUV轨迹跟踪与障碍物避障环境使用指南

## 概述

`AUVTrackingObstacleAvoidanceEnv` 是一个集成了障碍物避障、洋流干扰和轨迹跟踪功能的强化学习环境。该环境基于 `BaseMDP` 类构建，支持 Gymnasium API。

## 主要特性

### 1. 障碍物避障
- 自动生成随机位置的球形障碍物
- 基于三次样条插值的避障轨迹规划
- 实时碰撞检测
- 可配置的障碍物半径和安全距离

### 2. 洋流干扰
- 时变洋流模拟
- 可配置的洋流强度和变化率
- 洋流在NED坐标系和本体坐标系之间的转换

### 3. 轨迹跟踪
- Line-of-Sight (LOS) 制导算法
- 期望姿态计算
- 路径误差实时监测

### 4. 奖励机制
- 路径跟踪奖励
- 姿态跟踪奖励（俯仰角和偏航角）
- 避障奖励
- 碰撞惩罚
- 进步奖励

## 环境参数

### 观察空间 (22维)
```
[0-2]:   位置误差 (dx, dy, dz)
[3]:     路径误差 d_path
[4-8]:   速度 (u, v, w, q, r)
[9-12]:  期望俯仰角和误差的三角函数
[13-16]: 期望偏航角和误差的三角函数
[17-19]: 障碍物相对位置
[20]:    到障碍物的距离
[21]:    碰撞标志
```

### 动作空间 (3维)
```
[0]: 推力 (thrust) - [0, 1200] N
[1]: 垂直舵角 (rudder_v) - [-30, 30] deg
[2]: 水平舵角 (rudder_h) - [-30, 30] deg
```

## 快速开始

### 基础使用

```python
from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config

# 创建环境
config = get_config('with_obstacle')
env = AUVTrackingObstacleAvoidanceEnv(config)

# 重置环境
obs, info = env.reset(seed=42)

# 运行一个episode
done = False
total_reward = 0

while not done:
    # 选择动作（这里使用随机动作）
    action = env.action_space.sample()
    
    # 执行动作
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    done = terminated or truncated

print(f"Episode finished with total reward: {total_reward:.2f}")
env.close()
```

### 使用不同配置

#### 1. 无障碍物（仅轨迹跟踪）
```python
config = get_config('no_obstacle')
env = AUVTrackingObstacleAvoidanceEnv(config)
```

#### 2. 带障碍物（避障轨迹跟踪）
```python
config = get_config('with_obstacle')
env = AUVTrackingObstacleAvoidanceEnv(config)
```

#### 3. 带障碍物和洋流（完整挑战）
```python
config = get_config('with_current')
env = AUVTrackingObstacleAvoidanceEnv(config)
```

#### 4. 训练配置（启用模糊参数）
```python
config = get_config('training')
env = AUVTrackingObstacleAvoidanceEnv(config)
```

#### 5. 高难度配置
```python
config = get_config('hard')
env = AUVTrackingObstacleAvoidanceEnv(config)
```

### 自定义配置

```python
custom_config = {
    'max_steps': 500,
    'dt': 0.1,
    'obstacle_enabled': True,
    'r_obstacle': 5.0,           # 更大的障碍物
    'avoid_dis': 5.0,            # 更大的安全距离
    'current_enabled': True,
    'current_config': {
        'mu': 0.02,
        'Vmin': 0.2,
        'Vmax': 0.8,
        'Vc_init': 0.5,
        'alpha_init': 0.0,
        'beta_init': 0.0,
    },
    'reward_weights': {
        'path': 2.0,
        'pitch': 0.5,
        'yaw': 0.5,
        'obstacle': 3.0,
        'collision': 2.0,
    },
    'collision_penalty': -200.0,
}

env = AUVTrackingObstacleAvoidanceEnv(custom_config)
```

## 信息字典 (info)

环境的 `step()` 方法返回的 `info` 字典包含以下信息：

```python
info = {
    'state': {
        'position': [x, y, z],           # 当前位置
        'orientation': [phi, theta, psi], # 当前姿态
        'velocity': [u, v, w, p, q, r],  # 当前速度
    },
    'control': {
        'thrust': float,                  # 推力
        'rudder_v': float,                # 垂直舵角
        'rudder_h': float,                # 水平舵角
    },
    'desired': {
        'x': float, 'y': float, 'z': float,
        'pitch': float, 'yaw': float
    },
    'reward_info': {
        'path_error': float,
        'pitch_error': float,
        'yaw_error': float,
        'path_reward': float,
        'pitch_reward': float,
        'yaw_reward': float,
        'obstacle_reward': float,
        'collision_penalty': float,
        'progress_reward': float,
        'obstacle_distance': float,
        'collision_occurred': bool,
        'total_reward': float
    },
    'obstacle': {
        'enabled': bool,
        'distance': float,
        'collision': bool
    },
    'time': float,
    'step': int
}
```

## 训练示例

### 使用 Stable-Baselines3

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config

# 创建环境
config = get_config('training')
env = AUVTrackingObstacleAvoidanceEnv(config)

# 检查环境
check_env(env)

# 创建模型
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
)

# 训练模型
model.learn(total_timesteps=1_000_000)

# 保存模型
model.save("ppo_auv_obstacle_avoidance")

# 测试模型
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## 对比：新环境 vs 原始环境

### 与 AUV_env.py 的主要区别

| 特性 | AUV_env.py | AUVTrackingObstacleAvoidanceEnv |
|------|-----------|--------------------------------|
| API | 旧版 Gym | 新版 Gymnasium |
| 基类 | gym.Env | BaseMDP |
| 观察空间 | 22维 | 22维（结构优化） |
| 障碍物 | 支持 | 支持（增强版） |
| 洋流 | 支持 | 支持（可配置） |
| 碰撞检测 | 有限 | 完整实现 |
| 奖励函数 | 复杂 | 模块化可配置 |
| 配置管理 | 硬编码 | 配置文件管理 |
| 代码结构 | 单文件 | 模块化 |

### 与 auv_tracking_env.py 的主要区别

| 特性 | auv_tracking_env.py | AUVTrackingObstacleAvoidanceEnv |
|------|---------------------|--------------------------------|
| 轨迹生成 | RandomTraj3D | AvoidObstacles（带避障） |
| 观察空间 | 17维 | 22维（增加障碍物信息） |
| 障碍物 | 无 | 有 |
| 洋流 | 无 | 有 |
| 奖励机制 | 简单 | 复杂（包含避障） |

## 可视化

环境提供了基本的可视化功能（需要实现渲染器）：

```python
env.render()  # 渲染当前状态
```

也可以使用matplotlib手动绘制轨迹：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 收集轨迹数据
positions = []
obs, info = env.reset()

for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    positions.append(env.x[:3].flatten())
    if terminated or truncated:
        break

positions = np.array(positions)
ref_traj = info['reference_trajectory']

# 绘制3D轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ref_traj[0], ref_traj[1], ref_traj[2], 'b-', label='Reference')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', label='AUV')
ax.legend()
plt.show()
```

## 测试

运行测试脚本以验证环境：

```bash
python test_obstacle_avoidance_env.py
```

测试包括：
1. 基本功能测试
2. 碰撞检测测试
3. 洋流干扰测试
4. 轨迹可视化测试

## 常见问题

### Q1: 如何调整障碍物大小和位置？
A: 障碍物的位置是随机生成的。可以通过修改 `r_obstacle` 和 `avoid_dis` 参数来调整障碍物大小和安全距离。

### Q2: 如何禁用洋流？
A: 在配置中设置 `current_enabled: False`。

### Q3: 如何调整奖励权重？
A: 修改配置中的 `reward_weights` 字典。

### Q4: 环境支持向量化吗？
A: 可以使用 Stable-Baselines3 的 `DummyVecEnv` 或 `SubprocVecEnv` 进行向量化。

### Q5: 如何保存和加载环境状态？
A: 可以通过保存和恢复 `env.x`, `env.u`, `env.t` 等状态变量来实现。

## 扩展建议

1. **多障碍物支持**：扩展为支持多个障碍物
2. **动态障碍物**：添加移动障碍物
3. **更复杂的洋流模型**：实现涡流、湍流等
4. **传感器模拟**：添加噪声和传感器限制
5. **任务多样化**：添加不同类型的任务（巡航、探索等）

## 参考文献

- Line-of-Sight 制导算法
- AUV动力学模型
- 障碍物避障算法

## 联系方式

如有问题或建议，请联系开发团队。

