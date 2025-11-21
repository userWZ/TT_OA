"""
AUV轨迹跟踪与障碍物避障环境使用示例
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config


def example_1_basic_usage():
    """示例1：基础使用"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # 创建环境
    config = get_config('with_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Obstacle enabled: {info['obstacle_enabled']}")
    
    # 运行一个episode
    total_reward = 0
    for step in range(100):
        # 随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"Step {step}: reward={reward:.3f}, "
                  f"path_error={info['reward_info']['path_error']:.3f}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()


def example_2_simple_controller():
    """示例2：使用简单的比例控制器"""
    print("\n" + "=" * 60)
    print("Example 2: Simple Proportional Controller")
    print("=" * 60)
    
    config = get_config('with_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    obs, info = env.reset(seed=123)
    
    total_reward = 0
    path_errors = []
    
    for step in range(300):
        # 简单的比例控制器
        pos_error = obs[:3]  # 位置误差
        path_error = obs[3]   # 路径误差
        
        # 计算控制动作
        thrust = 400.0 if path_error > 2.0 else 300.0
        rudder_v = -np.clip(pos_error[2] * 0.15, -0.4, 0.4)
        rudder_h = -np.clip((pos_error[0] + pos_error[1]) * 0.1, -0.4, 0.4)
        
        action = np.array([thrust, rudder_v, rudder_h])
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        path_errors.append(info['reward_info']['path_error'])
        
        if step % 50 == 0:
            print(f"Step {step}: error={info['reward_info']['path_error']:.3f}, "
                  f"reward={reward:.3f}, collision={info['obstacle']['collision']}")
        
        if terminated or truncated:
            if 'termination_reason' in info:
                print(f"Terminated: {info['termination_reason']}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average path error: {np.mean(path_errors):.3f}")
    print(f"Final path error: {path_errors[-1]:.3f}")
    
    env.close()


def example_3_compare_with_without_obstacle():
    """示例3：对比有无障碍物的性能"""
    print("\n" + "=" * 60)
    print("Example 3: Compare With/Without Obstacle")
    print("=" * 60)
    
    # 创建两个环境
    config_no_obs = get_config('no_obstacle')
    config_with_obs = get_config('with_obstacle')
    
    env_no_obs = AUVTrackingObstacleAvoidanceEnv(config_no_obs)
    env_with_obs = AUVTrackingObstacleAvoidanceEnv(config_with_obs)
    
    # 使用相同的种子
    seed = 42
    np.random.seed(seed)
    
    # 测试无障碍物环境
    obs1, _ = env_no_obs.reset(seed=seed)
    reward_no_obs = []
    for _ in range(200):
        action = env_no_obs.action_space.sample()
        obs1, reward, term, trunc, _ = env_no_obs.step(action)
        reward_no_obs.append(reward)
        if term or trunc:
            break
    
    # 测试有障碍物环境
    np.random.seed(seed)
    obs2, _ = env_with_obs.reset(seed=seed)
    reward_with_obs = []
    for _ in range(200):
        action = env_with_obs.action_space.sample()
        obs2, reward, term, trunc, _ = env_with_obs.step(action)
        reward_with_obs.append(reward)
        if term or trunc:
            break
    
    print(f"No obstacle  - Avg reward: {np.mean(reward_no_obs):.3f}, "
          f"Total: {np.sum(reward_no_obs):.2f}")
    print(f"With obstacle - Avg reward: {np.mean(reward_with_obs):.3f}, "
          f"Total: {np.sum(reward_with_obs):.2f}")
    
    env_no_obs.close()
    env_with_obs.close()


def example_4_visualize_trajectory():
    """示例4：可视化轨迹"""
    print("\n" + "=" * 60)
    print("Example 4: Visualize Trajectory")
    print("=" * 60)
    
    config = get_config('with_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    obs, info = env.reset(seed=42)
    
    # 收集数据
    auv_positions = [env.x[:3].flatten()]
    ref_trajectory = env.ref_traj
    rewards = []
    path_errors = []
    obstacle_distances = []
    
    print("Running simulation...")
    for step in range(400):
        # 简单控制器
        pos_error = obs[:3]
        thrust = 350.0
        rudder_v = -np.clip(pos_error[2] * 0.12, -0.35, 0.35)
        rudder_h = -np.clip((pos_error[0] + pos_error[1]) * 0.08, -0.35, 0.35)
        
        action = np.array([thrust, rudder_v, rudder_h])
        obs, reward, terminated, truncated, info = env.step(action)
        
        auv_positions.append(env.x[:3].flatten())
        rewards.append(reward)
        path_errors.append(info['reward_info']['path_error'])
        obstacle_distances.append(info['reward_info']['obstacle_distance'])
        
        if terminated or truncated:
            break
    
    auv_positions = np.array(auv_positions)
    
    # 创建图形
    fig = plt.figure(figsize=(18, 10))
    
    # 3D轨迹
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(ref_trajectory[0], ref_trajectory[1], ref_trajectory[2], 
             'b-', linewidth=2, alpha=0.6, label='Reference')
    ax1.plot(auv_positions[:, 0], auv_positions[:, 1], auv_positions[:, 2], 
             'r-', linewidth=2, label='AUV')
    
    # 绘制障碍物
    if env.obstacle is not None:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = env.obstacle.position[0] + env.obstacle.radius*np.cos(u)*np.sin(v)
        y = env.obstacle.position[1] + env.obstacle.radius*np.sin(u)*np.sin(v)
        z = env.obstacle.position[2] + env.obstacle.radius*np.cos(v)
        ax1.plot_surface(x, y, z, alpha=0.3, color='gray')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.invert_zaxis()
    
    # 俯视图
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(ref_trajectory[0], ref_trajectory[1], 'b-', linewidth=2, label='Reference')
    ax2.plot(auv_positions[:, 0], auv_positions[:, 1], 'r-', linewidth=2, label='AUV')
    if env.obstacle is not None:
        circle = plt.Circle((env.obstacle.position[0], env.obstacle.position[1]), 
                           env.obstacle.radius, color='gray', alpha=0.5, label='Obstacle')
        ax2.add_patch(circle)
        # 安全区域
        safety_circle = plt.Circle((env.obstacle.position[0], env.obstacle.position[1]), 
                                  env.obstacle.radius + env.config['avoid_dis'], 
                                  color='yellow', alpha=0.2, linestyle='--', fill=False)
        ax2.add_patch(safety_circle)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 侧视图
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(ref_trajectory[0], ref_trajectory[2], 'b-', linewidth=2, label='Reference')
    ax3.plot(auv_positions[:, 0], auv_positions[:, 2], 'r-', linewidth=2, label='AUV')
    if env.obstacle is not None:
        circle = plt.Circle((env.obstacle.position[0], env.obstacle.position[2]), 
                           env.obstacle.radius, color='gray', alpha=0.5)
        ax3.add_patch(circle)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    # 奖励曲线
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(rewards, linewidth=1.5)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Reward')
    ax4.set_title('Reward over Time')
    ax4.grid(True, alpha=0.3)
    
    # 路径误差
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(path_errors, linewidth=1.5, color='orange')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Path Error (m)')
    ax5.set_title('Path Tracking Error')
    ax5.grid(True, alpha=0.3)
    
    # 障碍物距离
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(obstacle_distances, linewidth=1.5, color='green')
    if env.obstacle is not None:
        ax6.axhline(y=env.obstacle.radius, color='r', linestyle='--', 
                   label='Collision boundary', alpha=0.7)
        ax6.axhline(y=env.obstacle.radius + env.config['avoid_dis'], 
                   color='orange', linestyle='--', 
                   label='Safety boundary', alpha=0.7)
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Distance to Obstacle (m)')
    ax6.set_title('Obstacle Distance')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = 'example_trajectory_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    # 打印统计信息
    print(f"\nStatistics:")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Total reward: {np.sum(rewards):.2f}")
    print(f"  Average path error: {np.mean(path_errors):.3f}")
    print(f"  Final path error: {path_errors[-1]:.3f}")
    print(f"  Min obstacle distance: {np.min(obstacle_distances):.3f}")
    print(f"  Collision occurred: {info['obstacle']['collision']}")
    
    env.close()


def example_5_different_configs():
    """示例5：测试不同配置"""
    print("\n" + "=" * 60)
    print("Example 5: Test Different Configurations")
    print("=" * 60)
    
    configs_to_test = [
        ('no_obstacle', 'No Obstacle'),
        ('with_obstacle', 'With Obstacle'),
        ('with_current', 'With Obstacle + Ocean Current'),
    ]
    
    results = {}
    
    for config_name, description in configs_to_test:
        print(f"\nTesting: {description}")
        
        config = get_config(config_name)
        env = AUVTrackingObstacleAvoidanceEnv(config)
        
        # 运行多个episodes
        episode_rewards = []
        for episode in range(5):
            obs, info = env.reset(seed=episode)
            total_reward = 0
            
            for _ in range(300):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
        
        results[description] = {
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'min': np.min(episode_rewards),
            'max': np.max(episode_rewards)
        }
        
        print(f"  Mean reward: {results[description]['mean']:.2f} ± {results[description]['std']:.2f}")
        print(f"  Range: [{results[description]['min']:.2f}, {results[description]['max']:.2f}]")
        
        env.close()
    
    # 打印汇总
    print("\n" + "-" * 60)
    print("Summary:")
    for desc, res in results.items():
        print(f"{desc:40s}: {res['mean']:8.2f} ± {res['std']:6.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AUV Obstacle Avoidance Environment Examples")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_usage()
        example_2_simple_controller()
        example_3_compare_with_without_obstacle()
        example_4_visualize_trajectory()
        example_5_different_configs()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()

