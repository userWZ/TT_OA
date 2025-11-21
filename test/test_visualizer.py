"""
Test the trajectory visualizer with sample data
"""
import numpy as np
import matplotlib.pyplot as plt
from env.utils.trajectory_visualizer import plot_3d_trajectory, plot_multiple_trajectories

# Generate sample trajectory data
n_points = 100
t = np.linspace(0, 10, n_points)

# Actual trajectory (spiral)
actual_traj = np.column_stack([
    10 * np.cos(t),
    10 * np.sin(t),
    -0.5 * t
])

# Reference trajectory (3xN format, like in the environment)
ref_traj = np.array([
    10 * np.cos(t) + 1,  # x with offset
    10 * np.sin(t) + 1,  # y with offset
    -0.5 * t + 0.5       # z with offset
])

# Obstacle data
obstacle_pos = np.array([5.0, 5.0, -2.5])
obstacle_radius = 2.0
safety_distance = 1.0

print("Testing single trajectory plot...")
print(f"Actual trajectory shape: {actual_traj.shape}")
print(f"Reference trajectory shape: {ref_traj.shape}")
print(f"Obstacle position: {obstacle_pos}")
print(f"Obstacle radius: {obstacle_radius}")

# Test 1: Single trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

plot_3d_trajectory(
    ax=ax,
    actual_trajectory=actual_traj,
    reference_trajectory=ref_traj,
    obstacle_position=obstacle_pos,
    obstacle_radius=obstacle_radius,
    safety_distance=safety_distance,
    show_start_end=True,
    show_legend=True,
    collision_occurred=False,
    title="Test Trajectory with Obstacle"
)

plt.savefig('test_single_trajectory.png', dpi=150, bbox_inches='tight')
print("Saved: test_single_trajectory.png")
plt.close()

# Test 2: Multiple trajectories
print("\nTesting multiple trajectories plot...")
trajectories = []
ref_trajs = []
obstacles = []

for i in range(3):
    offset = i * 2
    actual = np.column_stack([
        10 * np.cos(t) + offset,
        10 * np.sin(t) + offset,
        -0.5 * t
    ])
    ref = np.array([
        10 * np.cos(t) + 1 + offset,
        10 * np.sin(t) + 1 + offset,
        -0.5 * t + 0.5
    ])
    obs = {
        'position': np.array([5.0 + offset, 5.0 + offset, -2.5]),
        'radius': 2.0,
        'safety_distance': 1.0,
        'collision_occurred': i == 1  # Mark second one as collision
    }
    
    trajectories.append(actual)
    ref_trajs.append(ref)
    obstacles.append(obs)

fig = plot_multiple_trajectories(
    trajectories=trajectories,
    reference_trajectories=ref_trajs,
    obstacle_data=obstacles,
    titles=[f"Episode {i+1}" for i in range(3)],
    suptitle="Test Multiple Trajectories",
    save_path='test_multiple_trajectories.png'
)
print("Saved: test_multiple_trajectories.png")
plt.close()

print("\n✅ Visualizer test completed successfully!")
print("Check the generated PNG files to verify the plots.")

