"""
AUV轨迹跟踪与障碍物避障环境配置文件
"""

# 基础环境配置
BASE_CONFIG = {
    # 仿真参数
    'max_steps': 1000,
    'stop_steps': 1000,
    'dt': 0.1,
    
    # 初始状态设置
    'random_start': True,
    'random_seed': None,
    
    # 模糊参数
    'use_fuzzy': False,
}

# 障碍物配置
OBSTACLE_CONFIG = {
    'obstacle_enabled': True,
    'r_obstacle': 3.0,          # 障碍物半径
    'avoid_dis': 3.0,           # 避障安全距离
    'clip_parts': 5,            # 避障轨迹分段数
}

# 洋流配置
OCEAN_CURRENT_CONFIG = {
    'current_enabled': False,    # 是否启用洋流
    'current_config': {
        'mu': 0.01,             # 洋流变化率（0表示恒定洋流）
        'Vmin': 0.2,            # 最小洋流速度
        'Vmax': 0.6,            # 最大洋流速度
        'Vc_init': 0.3,         # 初始洋流速度
        'alpha_init': 0.0,      # 初始水平角度
        'beta_init': 0.0,       # 初始垂直角度
    }
}

# 奖励函数配置
REWARD_CONFIG = {
    'collision_penalty': -100.0,
    'safe_distance_reward_weight': 1.0,
    'reward_weights': {
        'path': 1.0,            # 路径跟踪权重
        'pitch': 0.5,           # 俯仰角权重
        'yaw': 0.5,             # 偏航角权重
        'obstacle': 2.0,        # 避障权重
        'collision': 1.0,       # 碰撞惩罚权重
    }
}

# ============================================
# 预定义配置组合
# ============================================

# 1. 无障碍物基础配置（仅轨迹跟踪）
CONFIG_NO_OBSTACLE = {
    **BASE_CONFIG,
    'obstacle_enabled': False,
    'current_enabled': False,
    'reward_weights': {
        'path': 1.0,
        'pitch': 0.5,
        'yaw': 0.5,
        'obstacle': 0.0,
        'collision': 0.0,
    }
}

# 2. 带障碍物无洋流配置（避障轨迹跟踪）
CONFIG_WITH_OBSTACLE = {
    **BASE_CONFIG,
    **OBSTACLE_CONFIG,
    'current_enabled': False,
    **REWARD_CONFIG,
}

# 3. 带障碍物和洋流配置（完整挑战）
CONFIG_WITH_OBSTACLE_AND_CURRENT = {
    **BASE_CONFIG,
    **OBSTACLE_CONFIG,
    **OCEAN_CURRENT_CONFIG,
    'current_enabled': True,
    **REWARD_CONFIG,
}

# 4. 训练配置（启用模糊参数）
CONFIG_TRAINING = {
    **BASE_CONFIG,
    **OBSTACLE_CONFIG,
    'use_fuzzy': True,          # 启用模糊参数以增强鲁棒性
    'current_enabled': False,
    **REWARD_CONFIG,
}

# 5. 高难度配置（障碍物+洋流+模糊参数）
CONFIG_HARD = {
    **BASE_CONFIG,
    **OBSTACLE_CONFIG,
    **OCEAN_CURRENT_CONFIG,
    'use_fuzzy': True,
    'current_enabled': True,
    'current_config': {
        'mu': 0.02,             # 更大的洋流变化率
        'Vmin': 0.3,
        'Vmax': 0.8,            # 更强的洋流
        'Vc_init': 0.5,
        'alpha_init': 0.0,
        'beta_init': 0.0,
    },
    **REWARD_CONFIG,
    'reward_weights': {
        'path': 1.5,            # 增加路径跟踪权重
        'pitch': 0.5,
        'yaw': 0.5,
        'obstacle': 3.0,        # 增加避障权重
        'collision': 2.0,       # 增加碰撞惩罚
    }
}

# 默认配置
DEFAULT_CONFIG = CONFIG_WITH_OBSTACLE


def get_config(config_name='default'):
    """
    获取指定的配置
    
    Args:
        config_name: 配置名称，可选值：
            - 'default': 默认配置（带障碍物无洋流）
            - 'no_obstacle': 无障碍物配置
            - 'with_obstacle': 带障碍物配置
            - 'with_current': 带障碍物和洋流配置
            - 'training': 训练配置
            - 'hard': 高难度配置
    
    Returns:
        config: 配置字典
    """
    configs = {
        'default': CONFIG_WITH_OBSTACLE,
        'no_obstacle': CONFIG_NO_OBSTACLE,
        'with_obstacle': CONFIG_WITH_OBSTACLE,
        'with_current': CONFIG_WITH_OBSTACLE_AND_CURRENT,
        'training': CONFIG_TRAINING,
        'hard': CONFIG_HARD,
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. "
                        f"Available configs: {list(configs.keys())}")
    
    return configs[config_name].copy()


if __name__ == "__main__":
    # 打印所有配置
    print("=" * 60)
    print("AUV Obstacle Avoidance Environment Configurations")
    print("=" * 60)
    
    for name in ['no_obstacle', 'with_obstacle', 'with_current', 'training', 'hard']:
        print(f"\n{name.upper()}:")
        config = get_config(name)
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

