"""
Trajectory Visualization Utilities

Provides reusable functions for visualizing AUV trajectories with obstacles.
Can be used in both real-time rendering and post-evaluation visualization.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Dict, Any, Union


def plot_3d_trajectory(
    ax: plt.Axes,
    actual_trajectory: np.ndarray,
    reference_trajectory: Optional[np.ndarray] = None,
    obstacle_position: Optional[np.ndarray] = None,
    obstacle_radius: Optional[float] = None,
    safety_distance: Optional[float] = None,
    show_start_end: bool = True,
    show_legend: bool = True,
    collision_occurred: bool = False,
    title: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plot a 3D trajectory with optional reference path and obstacles
    
    Args:
        ax: Matplotlib 3D axis to plot on
        actual_trajectory: Nx3 array of [x, y, z] positions
        reference_trajectory: Optional 3xM array or Nx3 array of reference path
        obstacle_position: Optional [x, y, z] position of obstacle center
        obstacle_radius: Optional radius of spherical obstacle
        safety_distance: Optional safety boundary around obstacle
        show_start_end: Whether to mark start/end points
        show_legend: Whether to show legend
        collision_occurred: Whether a collision occurred (changes obstacle color)
        title: Optional plot title
        **kwargs: Additional plot customization options
    """
    # Extract customization options
    ref_color = kwargs.get('ref_color', 'g')
    ref_style = kwargs.get('ref_style', '--')
    ref_linewidth = kwargs.get('ref_linewidth', 2)
    ref_alpha = kwargs.get('ref_alpha', 0.7)
    ref_label = kwargs.get('ref_label', 'Reference')
    
    traj_color = kwargs.get('traj_color', 'b')
    traj_style = kwargs.get('traj_style', '-')
    traj_linewidth = kwargs.get('traj_linewidth', 2.5)
    traj_alpha = kwargs.get('traj_alpha', 1.0)
    traj_label = kwargs.get('traj_label', 'Actual')
    
    start_color = kwargs.get('start_color', 'lime')
    start_marker = kwargs.get('start_marker', 'o')
    start_size = kwargs.get('start_size', 150)
    start_label = kwargs.get('start_label', 'Start')
    
    end_color = kwargs.get('end_color', 'red')
    end_marker = kwargs.get('end_marker', '*')
    end_size = kwargs.get('end_size', 200)
    end_label = kwargs.get('end_label', 'End')
    
    obstacle_color = kwargs.get('obstacle_color', 'red' if not collision_occurred else 'darkred')
    obstacle_alpha = kwargs.get('obstacle_alpha', 0.3 if not collision_occurred else 0.6)
    obstacle_label = kwargs.get('obstacle_label', 'Obstacle (Collision!)' if collision_occurred else 'Obstacle')
    
    safety_color = kwargs.get('safety_color', 'yellow')
    safety_alpha = kwargs.get('safety_alpha', 0.1)
    safety_label = kwargs.get('safety_label', 'Safety Zone')
    
    # Plot reference trajectory
    if reference_trajectory is not None and reference_trajectory.size > 0:
        try:
            if reference_trajectory.shape[0] == 3 and len(reference_trajectory.shape) == 2:
                # Format: 3xN (x_traj, y_traj, z_traj)
                ax.plot(reference_trajectory[0], reference_trajectory[1], reference_trajectory[2],
                       color=ref_color, linestyle=ref_style, linewidth=ref_linewidth, 
                       alpha=ref_alpha, label=ref_label)
            elif len(reference_trajectory.shape) == 2 and reference_trajectory.shape[1] == 3:
                # Format: Nx3
                ax.plot(reference_trajectory[:, 0], reference_trajectory[:, 1], reference_trajectory[:, 2],
                       color=ref_color, linestyle=ref_style, linewidth=ref_linewidth,
                       alpha=ref_alpha, label=ref_label)
            else:
                print(f"Warning: Unexpected reference trajectory shape: {reference_trajectory.shape}")
        except Exception as e:
            print(f"Warning: Failed to plot reference trajectory: {e}")
    
    # Plot actual trajectory
    if actual_trajectory.shape[1] == 3:
        ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2],
               color=traj_color, linestyle=traj_style, linewidth=traj_linewidth,
               alpha=traj_alpha, label=traj_label, zorder=5)
    
    # Mark start and end points
    if show_start_end and len(actual_trajectory) > 0:
        ax.scatter(actual_trajectory[0, 0], actual_trajectory[0, 1], actual_trajectory[0, 2],
                  c=start_color, s=start_size, marker=start_marker,
                  edgecolors='black', linewidths=2, label=start_label, zorder=10)
        ax.scatter(actual_trajectory[-1, 0], actual_trajectory[-1, 1], actual_trajectory[-1, 2],
                  c=end_color, s=end_size, marker=end_marker,
                  edgecolors='black', linewidths=2, label=end_label, zorder=10)
    
    # Plot obstacle
    has_obstacle = False
    if obstacle_position is not None and obstacle_radius is not None:
        has_obstacle = True
        _draw_sphere(ax, obstacle_position, obstacle_radius, 
                    color=obstacle_color, alpha=obstacle_alpha,
                    edgecolor='darkred', linewidth=0.5)
        
        # Draw obstacle center
        ax.scatter(obstacle_position[0], obstacle_position[1], obstacle_position[2],
                  c='darkred', s=50, marker='x', zorder=10)
    
    # Plot safety zone
    has_safety_zone = False
    if obstacle_position is not None and obstacle_radius is not None and safety_distance is not None:
        has_safety_zone = True
        safety_radius = obstacle_radius + safety_distance
        _draw_sphere(ax, obstacle_position, safety_radius,
                    color=safety_color, alpha=safety_alpha,
                    edgecolor='orange', linewidth=0.3)
    
    # Set labels
    ax.set_xlabel('X (m)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Y (m)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Z (m)', fontsize=10, fontweight='bold')
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Create legend with proxy artists (to avoid matplotlib 3D surface bug)
    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], color=traj_color, linestyle=traj_style, 
                      linewidth=traj_linewidth, label=traj_label)
        ]
        
        if reference_trajectory is not None:
            legend_elements.insert(0, plt.Line2D([0], [0], color=ref_color, 
                                                 linestyle=ref_style, linewidth=ref_linewidth, 
                                                 label=ref_label))
        
        if show_start_end:
            legend_elements.extend([
                plt.Line2D([0], [0], marker=start_marker, color='w', 
                          markerfacecolor=start_color, markeredgecolor='black',
                          markeredgewidth=2, markersize=10, label=start_label),
                plt.Line2D([0], [0], marker=end_marker, color='w',
                          markerfacecolor=end_color, markeredgecolor='black',
                          markeredgewidth=2, markersize=12, label=end_label)
            ])
        
        if has_obstacle:
            legend_elements.append(Patch(facecolor=obstacle_color, alpha=obstacle_alpha,
                                        edgecolor='darkred', label=obstacle_label))
        
        if has_safety_zone:
            legend_elements.append(Patch(facecolor=safety_color, alpha=safety_alpha,
                                        edgecolor='orange', label=safety_label))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.invert_zaxis()
    
    # Set equal aspect ratio for better visualization (considering all elements)
    if len(actual_trajectory) > 0:
        _set_equal_aspect_3d(
            ax, 
            actual_trajectory,
            reference_trajectory=reference_trajectory,
            obstacle_position=obstacle_position,
            obstacle_radius=obstacle_radius,
            safety_distance=safety_distance
        )


def plot_multiple_trajectories(
    trajectories: List[np.ndarray],
    reference_trajectories: Optional[List[np.ndarray]] = None,
    obstacle_data: Optional[List[Dict[str, Any]]] = None,
    titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6),
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot multiple trajectories in a grid
    
    Args:
        trajectories: List of Nx3 trajectory arrays
        reference_trajectories: Optional list of reference trajectories
        obstacle_data: Optional list of obstacle info dicts with 'position', 'radius', 'safety_distance'
        titles: Optional list of subplot titles
        suptitle: Optional main figure title
        figsize: Figure size (width, height)
        save_path: Optional path to save figure
        **kwargs: Additional plot customization options passed to plot_3d_trajectory
        
    Returns:
        Matplotlib figure object
    """
    n_plots = len(trajectories)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=figsize)
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    
    for i in range(n_plots):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        
        # Get reference trajectory if available
        ref_traj = None
        if reference_trajectories and i < len(reference_trajectories):
            ref_traj = reference_trajectories[i]
        
        # Get obstacle data if available
        obs_pos = None
        obs_radius = None
        safety_dist = None
        collision = False
        
        if obstacle_data and i < len(obstacle_data) and obstacle_data[i]:
            obs_info = obstacle_data[i]
            obs_pos = obs_info.get('position')
            obs_radius = obs_info.get('radius')
            safety_dist = obs_info.get('safety_distance')
            collision = obs_info.get('collision_occurred', False)
        
        # Get title
        title = None
        if titles and i < len(titles):
            title = titles[i]
        
        # Plot trajectory
        plot_3d_trajectory(
            ax, trajectories[i],
            reference_trajectory=ref_traj,
            obstacle_position=obs_pos,
            obstacle_radius=obs_radius,
            safety_distance=safety_dist,
            collision_occurred=collision,
            title=title,
            **kwargs
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectories plot saved to: {save_path}")
    
    return fig


def _draw_sphere(
    ax: plt.Axes,
    center: np.ndarray,
    radius: float,
    color: str = 'red',
    alpha: float = 0.3,
    edgecolor: str = 'darkred',
    linewidth: float = 0.5,
    resolution: Tuple[int, int] = (30, 20)
) -> Any:
    """
    Draw a sphere on a 3D axis
    
    Args:
        ax: Matplotlib 3D axis
        center: [x, y, z] center position
        radius: Sphere radius
        color: Surface color
        alpha: Transparency
        edgecolor: Edge color
        linewidth: Edge line width
        resolution: (u_resolution, v_resolution) for mesh grid
        
    Returns:
        Surface plot object
    """
    u = np.linspace(0, 2 * np.pi, resolution[0])
    v = np.linspace(0, np.pi, resolution[1])
    
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    return ax.plot_surface(x, y, z, color=color, alpha=alpha,
                          edgecolor=edgecolor, linewidth=linewidth,
                          antialiased=True)


def _set_equal_aspect_3d(
    ax: plt.Axes, 
    trajectory: np.ndarray,
    reference_trajectory: Optional[np.ndarray] = None,
    obstacle_position: Optional[np.ndarray] = None,
    obstacle_radius: Optional[float] = None,
    safety_distance: Optional[float] = None
) -> None:
    """
    Set equal aspect ratio for 3D plot, considering all elements
    
    Args:
        ax: Matplotlib 3D axis
        trajectory: Nx3 trajectory array
        reference_trajectory: Optional reference trajectory (3xN or Nx3)
        obstacle_position: Optional obstacle center position
        obstacle_radius: Optional obstacle radius
        safety_distance: Optional safety distance around obstacle
    """
    # Collect all points to determine bounds
    all_points = [trajectory]
    
    # Add reference trajectory points if available
    if reference_trajectory is not None and reference_trajectory.size > 0:
        if reference_trajectory.shape[0] == 3 and len(reference_trajectory.shape) == 2:
            # Format: 3xN, transpose to Nx3
            all_points.append(reference_trajectory.T)
        elif len(reference_trajectory.shape) == 2 and reference_trajectory.shape[1] == 3:
            # Format: Nx3
            all_points.append(reference_trajectory)
    
    # Add obstacle bounds if available
    if obstacle_position is not None and obstacle_radius is not None:
        # Calculate obstacle extent including safety zone
        total_radius = obstacle_radius
        if safety_distance is not None:
            total_radius += safety_distance
        
        # Add 8 corner points of the obstacle's bounding box
        obstacle_bounds = np.array([
            [obstacle_position[0] - total_radius, obstacle_position[1] - total_radius, obstacle_position[2] - total_radius],
            [obstacle_position[0] + total_radius, obstacle_position[1] - total_radius, obstacle_position[2] - total_radius],
            [obstacle_position[0] - total_radius, obstacle_position[1] + total_radius, obstacle_position[2] - total_radius],
            [obstacle_position[0] + total_radius, obstacle_position[1] + total_radius, obstacle_position[2] - total_radius],
            [obstacle_position[0] - total_radius, obstacle_position[1] - total_radius, obstacle_position[2] + total_radius],
            [obstacle_position[0] + total_radius, obstacle_position[1] - total_radius, obstacle_position[2] + total_radius],
            [obstacle_position[0] - total_radius, obstacle_position[1] + total_radius, obstacle_position[2] + total_radius],
            [obstacle_position[0] + total_radius, obstacle_position[1] + total_radius, obstacle_position[2] + total_radius],
        ])
        all_points.append(obstacle_bounds)
    
    # Combine all points
    combined_points = np.vstack(all_points)
    
    # Calculate bounds
    x_min, y_min, z_min = combined_points.min(axis=0)
    x_max, y_max, z_max = combined_points.max(axis=0)
    
    # Calculate center and max range
    mid_x = (x_max + x_min) * 0.5
    mid_y = (y_max + y_min) * 0.5
    mid_z = (z_max + z_min) * 0.5
    
    max_range = np.array([
        x_max - x_min,
        y_max - y_min,
        z_max - z_min
    ]).max() / 2.0
    
    # Add a small margin (5%)
    max_range *= 1.05
    
    # Set axis limits
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def create_evaluation_summary_plot(
    trajectories: List[np.ndarray],
    reference_trajectories: Optional[List[np.ndarray]],
    obstacle_data: Optional[List[Dict[str, Any]]],
    metrics: Dict[str, Any],
    algorithm: str,
    output_dir: str,
    **kwargs
) -> None:
    """
    Create comprehensive evaluation summary with trajectories and metrics
    
    Args:
        trajectories: List of trajectory arrays
        reference_trajectories: List of reference trajectories
        obstacle_data: List of obstacle information
        metrics: Dictionary of evaluation metrics
        algorithm: Algorithm name
        output_dir: Output directory for saving plots
        **kwargs: Additional plot customization
    """
    import os
    
    # Plot sample trajectories (up to 3)
    n_samples = min(3, len(trajectories))
    if n_samples > 0:
        sample_titles = [f'{algorithm.upper()} - Episode {i+1}' for i in range(n_samples)]
        
        trajectory_fig = plot_multiple_trajectories(
            trajectories[:n_samples],
            reference_trajectories[:n_samples] if reference_trajectories else None,
            obstacle_data[:n_samples] if obstacle_data else None,
            titles=sample_titles,
            suptitle=f'{algorithm.upper()} Evaluation - Sample Trajectories',
            save_path=os.path.join(output_dir, f'{algorithm}_sample_trajectories.png'),
            **kwargs
        )
        plt.close(trajectory_fig)
    
    # Plot metrics summary
    _plot_metrics_summary(metrics, algorithm, output_dir)


def _plot_metrics_summary(metrics: Dict[str, Any], algorithm: str, output_dir: str) -> None:
    """Plot evaluation metrics summary"""
    import os
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Reward distribution
    axes[0, 0].hist(metrics['episode_rewards'], bins=20, edgecolor='black', color='steelblue')
    axes[0, 0].axvline(metrics['mean_reward'], color='r', linestyle='--',
                      label=f"Mean: {metrics['mean_reward']:.2f}")
    axes[0, 0].set_xlabel('Episode Reward')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'{algorithm.upper()} - Reward Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode length
    axes[0, 1].hist(metrics['episode_lengths'], bins=20, edgecolor='black', color='orange')
    axes[0, 1].axvline(metrics['mean_length'], color='r', linestyle='--',
                      label=f"Mean: {metrics['mean_length']:.1f}")
    axes[0, 1].set_xlabel('Episode Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Episode Length Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Final error
    if metrics['final_errors']:
        axes[1, 0].hist(metrics['final_errors'], bins=20, edgecolor='black', color='green')
        axes[1, 0].axvline(metrics['mean_final_error'], color='r', linestyle='--',
                          label=f"Mean: {metrics['mean_final_error']:.3f}")
        axes[1, 0].set_xlabel('Final Path Error (m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Final Error Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Success/Collision rate
    success_rate = (1 - metrics['collision_rate']) * 100
    axes[1, 1].bar(['Success', 'Collision'],
                  [success_rate, metrics['collision_rate'] * 100],
                  color=['green', 'red'], edgecolor='black')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_title('Success vs Collision Rate')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate([success_rate, metrics['collision_rate'] * 100]):
        axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{algorithm}_evaluation_metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Metrics plot saved to: {plot_path}")
    plt.close()

