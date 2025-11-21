"""
测试AUV轨迹跟踪与障碍物避障环境
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    # 创建环境（带障碍物）
    config = get_config('with_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # 重置环境
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Obstacle enabled: {info['obstacle_enabled']}")
    print(f"Current enabled: {info['current_enabled']}")
    
    if info['obstacle_info'] is not None:
        obstacle_center = info['obstacle_info'][:3]
        print(f"Obstacle center: [{obstacle_center[0]:.2f}, {obstacle_center[1]:.2f}, {obstacle_center[2]:.2f}]")
        print(f"Obstacle radius: {info['obstacle_info'][3]:.2f}")
        print(f"Avoidance distance: {info['obstacle_info'][4]:.2f}")
    
    # 执行几步
    print("\nRunning 10 steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 5 == 0:
            print(f"Step {i}: reward={reward:.3f}, "
                  f"path_error={info['reward_info']['path_error']:.3f}, "
                  f"obstacle_distance={info['reward_info']['obstacle_distance']:.3f}")
        
        if terminated or truncated:
            print(f"Episode terminated at step {i}")
            break
    
    env.close()
    print("\n✓ Basic functionality test passed!")


def test_collision_detection():
    """测试碰撞检测"""
    print("\n" + "=" * 60)
    print("Test 2: Collision Detection")
    print("=" * 60)
    
    config = get_config('with_obstacle')
    config['collision_penalty'] = -200.0  # 增大碰撞惩罚以便观察
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    obs, info = env.reset(seed=123)
    
    if info['obstacle_info'] is not None:
        obstacle_center = info['obstacle_info'][:3]
        print(f"Obstacle position: {obstacle_center}")
        
        # 手动设置AUV位置接近障碍物
        env.x[0] = obstacle_center[0]
        env.x[1] = obstacle_center[1]
        env.x[2] = obstacle_center[2]
        
        # 执行一步
        action = np.array([300.0, 0.0, 0.0])  # 前进
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Distance to obstacle: {info['reward_info']['obstacle_distance']:.3f}")
        print(f"Collision occurred: {info['reward_info']['collision_occurred']}")
        print(f"Collision penalty: {info['reward_info']['collision_penalty']:.3f}")
        print(f"Terminated: {terminated}")
        
        if info['reward_info']['collision_occurred']:
            print("\n✓ Collision detection working!")
        else:
            print("\n⚠ Collision not detected (AUV might be just outside collision radius)")
    else:
        print("⚠ No obstacle generated in this episode")
    
    env.close()


def test_ocean_current():
    """测试洋流干扰"""
    print("\n" + "=" * 60)
    print("Test 3: Ocean Current Disturbance")
    print("=" * 60)
    
    # 创建两个环境：一个有洋流，一个没有
    config_no_current = get_config('with_obstacle')
    config_no_current['current_enabled'] = False
    
    config_with_current = get_config('with_current')
    config_with_current['current_config']['Vc_init'] = 0.5  # 设置较大的洋流
    
    env_no_current = AUVTrackingObstacleAvoidanceEnv(config_no_current)
    env_with_current = AUVTrackingObstacleAvoidanceEnv(config_with_current)
    
    # 使用相同的种子初始化
    obs1, _ = env_no_current.reset(seed=42)
    obs2, _ = env_with_current.reset(seed=42)
    
    # 执行相同的动作序列
    np.random.seed(42)
    positions_no_current = [env_no_current.x[:3].copy()]
    positions_with_current = [env_with_current.x[:3].copy()]
    
    for i in range(50):
        action = env_no_current.action_space.sample()
        
        obs1, _, term1, trunc1, _ = env_no_current.step(action)
        obs2, _, term2, trunc2, _ = env_with_current.step(action)
        
        positions_no_current.append(env_no_current.x[:3].copy())
        positions_with_current.append(env_with_current.x[:3].copy())
        
        if term1 or trunc1 or term2 or trunc2:
            break
    
    # 计算轨迹差异
    positions_no_current = np.array(positions_no_current).squeeze()
    positions_with_current = np.array(positions_with_current).squeeze()
    
    diff = np.linalg.norm(positions_with_current - positions_no_current, axis=1)
    avg_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"Average position difference: {avg_diff:.3f}")
    print(f"Maximum position difference: {max_diff:.3f}")
    
    if avg_diff > 0.1:  # 应该有显著差异
        print("\n✓ Ocean current effect detected!")
    else:
        print("\n⚠ Ocean current effect might be too small")
    
    env_no_current.close()
    env_with_current.close()


def test_trajectory_visualization():
    """测试轨迹可视化"""
    print("\n" + "=" * 60)
    print("Test 4: Trajectory Visualization")
    print("=" * 60)
    
    config = get_config('with_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    obs, info = env.reset(seed=42)
    
    # 收集轨迹数据
    auv_positions = [env.x[:3].flatten()]
    ref_trajectory = info['reference_trajectory']
    
    # 简单的比例控制器
    print("Running episode with simple controller...")
    for i in range(200):
        # 简单的PD控制
        pos_error = obs[:3]
        velocity = obs[4:9]
        
        # 计算动作（简单的比例控制）
        thrust = 400.0 if np.linalg.norm(pos_error) > 1.0 else 300.0
        rudder_v = -np.clip(pos_error[2] * 0.1, -0.3, 0.3)
        rudder_h = -np.clip(pos_error[1] * 0.1, -0.3, 0.3)
        
        action = np.array([thrust, rudder_v, rudder_h])
        obs, reward, terminated, truncated, info = env.step(action)
        
        auv_positions.append(env.x[:3].flatten())
        
        if i % 50 == 0:
            print(f"  Step {i}: error={info['reward_info']['path_error']:.3f}, "
                  f"reward={reward:.3f}")
        
        if terminated or truncated:
            if 'termination_reason' in info:
                print(f"  Terminated: {info['termination_reason']}")
            break
    
    auv_positions = np.array(auv_positions)
    
    # 绘制3D轨迹
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1：3D视图
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(ref_trajectory[0], ref_trajectory[1], ref_trajectory[2], 
             'b-', label='Reference', linewidth=2, alpha=0.7)
    ax1.plot(auv_positions[:, 0], auv_positions[:, 1], auv_positions[:, 2], 
             'r-', label='AUV', linewidth=2)
    
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
    
    # 子图2：俯视图
    ax2 = fig.add_subplot(132)
    ax2.plot(ref_trajectory[0], ref_trajectory[1], 'b-', label='Reference', linewidth=2)
    ax2.plot(auv_positions[:, 0], auv_positions[:, 1], 'r-', label='AUV', linewidth=2)
    if env.obstacle is not None:
        circle = plt.Circle((env.obstacle.position[0], env.obstacle.position[1]), 
                           env.obstacle.radius, color='gray', alpha=0.5)
        ax2.add_patch(circle)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 子图3：侧视图
    ax3 = fig.add_subplot(133)
    ax3.plot(ref_trajectory[0], ref_trajectory[2], 'b-', label='Reference', linewidth=2)
    ax3.plot(auv_positions[:, 0], auv_positions[:, 2], 'r-', label='AUV', linewidth=2)
    if env.obstacle is not None:
        circle = plt.Circle((env.obstacle.position[0], env.obstacle.position[2]), 
                           env.obstacle.radius, color='gray', alpha=0.5)
        ax3.add_patch(circle)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View')
    ax3.legend()
    ax3.grid(True)
    ax3.invert_yaxis()
    
    plt.tight_layout()
    
    # 保存图像
    save_path = 'test_trajectory_obstacle_avoidance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Trajectory visualization saved to: {save_path}")
    
    # 显示图像（可选）
    # plt.show()
    
    env.close()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("AUV Obstacle Avoidance Environment Tests")
    print("=" * 60 + "\n")
    
    try:
        test_basic_functionality()
        test_collision_detection()
        test_ocean_current()
        test_trajectory_visualization()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()

