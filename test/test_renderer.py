"""
Quick test for optimized AUV renderer
"""
import numpy as np
import matplotlib.pyplot as plt

from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config


def test_renderer_with_obstacle():
    """Test renderer with obstacle avoidance environment"""
    print("=" * 60)
    print("Testing Renderer with Obstacle Avoidance")
    print("=" * 60)
    
    # Create environment with obstacle
    config = get_config('with_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    obs, info = env.reset(seed=42)
    
    print(f"\nEnvironment initialized:")
    print(f"  Has obstacle: {env.obstacle is not None}")
    if env.obstacle:
        print(f"  Obstacle position: {env.obstacle.position}")
        print(f"  Obstacle radius: {env.obstacle.radius}")
    
    print(f"\nRunning simulation with rendering...")
    print("  Close the plot window to continue...")
    
    # Run simulation with simple controller
    for step in range(50):
        # Simple proportional controller
        pos_error = obs[:3]
        thrust = 350.0
        rudder_v = -np.clip(pos_error[2] * 0.1, -0.3, 0.3)
        rudder_h = -np.clip((pos_error[0] + pos_error[1]) * 0.08, -0.3, 0.3)
        
        action = np.array([thrust, rudder_v, rudder_h])
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render every 10 steps
        if step % 10 == 0:
            env.render(
                show_3d=True,
                show_control=True,
                show_error=True,
                show_obstacle_info=True
            )
            print(f"  Step {step}: "
                  f"distance={info['reward_info']['obstacle_distance']:.2f}m, "
                  f"collision={info['reward_info']['collision_occurred']}")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step}")
            if 'termination_reason' in info:
                print(f"  Reason: {info['termination_reason']}")
            break
    
    # Final render
    print("\nFinal rendering...")
    env.render()
    plt.show()
    
    env.close()
    print("\n✓ Renderer test completed!")


def test_renderer_without_obstacle():
    """Test renderer with basic tracking environment"""
    print("\n" + "=" * 60)
    print("Testing Renderer without Obstacle")
    print("=" * 60)
    
    # Create environment without obstacle
    config = get_config('no_obstacle')
    env = AUVTrackingObstacleAvoidanceEnv(config)
    
    obs, info = env.reset(seed=42)
    
    print(f"\nEnvironment initialized:")
    print(f"  Has obstacle: {env.obstacle is not None}")
    
    print(f"\nRunning simulation...")
    
    # Run a few steps
    for step in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            env.render(
                show_3d=True,
                show_control=True,
                show_error=True
            )
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n✓ Renderer test completed!")


if __name__ == "__main__":
    print("\nAUV Renderer Optimization Test\n")
    
    try:
        # Test with obstacle
        test_renderer_with_obstacle()
        
        # Test without obstacle
        test_renderer_without_obstacle()
        
        print("\n" + "=" * 60)
        print("All renderer tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

