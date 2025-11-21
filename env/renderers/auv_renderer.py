"""
AUV Environment Renderer

Provides visualization for AUV trajectory tracking environments with optional
obstacle avoidance support.
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from typing import Any, Dict, Optional, Tuple


class AUVRenderer:
    """
    Universal AUV environment renderer
    
    Supports visualization for:
    - Basic trajectory tracking environments
    - Obstacle avoidance environments
    - Real-time trajectory updates
    - Control inputs and error monitoring
    
    Attributes:
        env: AUV environment instance
        has_obstacle: Whether environment has obstacle avoidance
        render_config: Rendering configuration dictionary
    """
    
    def __init__(self, env):
        """
        Initialize renderer
        
        Args:
            env: AUV environment instance (AUVTrackingEnv or AUVTrackingObstacleAvoidanceEnv)
        """
        self.env = env
        # Check if environment has obstacle (single obstacle object)
        self.has_obstacle = hasattr(env, 'obstacle') and env.obstacle is not None
        
        # Rendering configuration
        self.render_config = {
            'show_3d': True,              # 3D trajectory plot
            'show_control': True,         # Control inputs plot
            'show_error': True,           # Error tracking plot
            'show_obstacle_info': True,   # Obstacle distance info (if applicable)
            'figure_size': (15, 10)       # Figure size in inches
        }
        
        # Figure objects
        self.fig: Optional[plt.Figure] = None
        self.ax_3d: Optional[plt.Axes] = None
        self.ax_control: Optional[plt.Axes] = None
        self.ax_error: Optional[plt.Axes] = None
        self.ax_obstacle: Optional[plt.Axes] = None
        
        # Plot elements
        self.reference_line = None
        self.trajectory_line = None
        self.auv_position = None
        self.thrust_line = None
        self.rudder_v_line = None
        self.rudder_h_line = None
        self.path_error_line = None
        self.pitch_error_line = None
        self.yaw_error_line = None
        self.obstacle_distance_line = None
        
        # History data
        self.history_positions = []
        self.history_data: Optional[Dict] = None
        self._current_render_config: Optional[Dict] = None
        
        # Obstacle visualization
        self.obstacle_surface = None      # Obstacle surface plot
        self.safety_sphere_surface = None # Safety boundary surface
        self.collision_occurred = False   # Collision flag

    def _to_float(self, x: Any) -> float:
        """
        Convert any numeric type to float
        
        Args:
            x: Numeric value (tensor, numpy array, or scalar)
            
        Returns:
            Float value
        """
        if hasattr(x, 'item'):  # Handle tensors
            return float(x.item())
        elif isinstance(x, np.ndarray):  # Handle numpy arrays
            return float(x.flatten()[0] if x.size > 0 else 0.0)
        return float(x)

    def _limit_angle(self, angle: float) -> float:
        """
        Limit angle to [-pi, pi] range
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle in [-pi, pi]
        """
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _init_3d_plot(self):
        """Initialize 3D trajectory plot"""
        # Plot reference trajectory
        self.reference_line, = self.ax_3d.plot(
            self.env.x_traj, self.env.y_traj, self.env.z_traj,
            'g--', linewidth=2, alpha=0.7, label='Reference'
        )
        
        # Plot actual trajectory
        self.trajectory_line, = self.ax_3d.plot(
            [], [], [], 'b-', linewidth=2, label='Actual'
        )
        
        # Plot AUV position
        self.auv_position = self.ax_3d.scatter(
            [], [], [], c='red', marker='o', s=100, 
            edgecolors='black', linewidths=1.5, label='AUV'
        )
        
        # Draw obstacle if present
        if self.has_obstacle:
            obstacle = self.env.obstacle
            obs_pos = obstacle.position
            obs_radius = obstacle.radius
            
            # Generate sphere mesh
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 20)
            x = obs_pos[0] + obs_radius * np.outer(np.cos(u), np.sin(v))
            y = obs_pos[1] + obs_radius * np.outer(np.sin(u), np.sin(v))
            z = obs_pos[2] + obs_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Draw obstacle surface (without label, will add manually)
            self.obstacle_surface = self.ax_3d.plot_surface(
                x, y, z,
                color='red',
                alpha=0.3,
                edgecolor='darkred',
                linewidth=0.5,
                antialiased=True
            )
            
            # Draw safety boundary if avoidance distance is configured
            if hasattr(self.env.config, '__getitem__') and 'avoid_dis' in self.env.config:
                safety_radius = obs_radius + self.env.config['avoid_dis']
                x_safe = obs_pos[0] + safety_radius * np.outer(np.cos(u), np.sin(v))
                y_safe = obs_pos[1] + safety_radius * np.outer(np.sin(u), np.sin(v))
                z_safe = obs_pos[2] + safety_radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                self.safety_sphere_surface = self.ax_3d.plot_surface(
                    x_safe, y_safe, z_safe,
                    color='yellow',
                    alpha=0.1,
                    edgecolor='orange',
                    linewidth=0.3,
                    linestyle='--',
                    antialiased=True
                )
        
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('AUV Trajectory Tracking with Obstacle Avoidance')
        
        # Create legend manually to avoid matplotlib surface legend issues
        legend_elements = [
            plt.Line2D([0], [0], color='g', linestyle='--', linewidth=2, label='Reference'),
            plt.Line2D([0], [0], color='b', linewidth=2, label='Actual'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=8, label='AUV', markeredgecolor='black')
        ]
        
        if self.has_obstacle:
            # Add obstacle to legend using a proxy artist
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor='red', edgecolor='darkred', alpha=0.3, label='Obstacle')
            )
            if hasattr(self.env.config, '__getitem__') and 'avoid_dis' in self.env.config:
                legend_elements.append(
                    Patch(facecolor='yellow', edgecolor='orange', alpha=0.1, label='Safety Zone')
                )
        
        self.ax_3d.legend(handles=legend_elements, loc='upper right')
        self.ax_3d.grid(True, alpha=0.3)
        
        # Add info text
        info_text = f'Step: {self.env._count}'
        if self.has_obstacle:
            info_text += '\nCollision: No'
        
        self.step_text = self.ax_3d.text2D(
            0.02, 0.98,
            info_text,
            transform=self.ax_3d.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            zorder=10
        )

    def _init_control_plot(self):
        """Initialize control inputs plot"""
        self.thrust_line, = self.ax_control.plot(
            [], [], 'r-', linewidth=2, label='Thrust (N)'
        )
        self.rudder_v_line, = self.ax_control.plot(
            [], [], 'g-', linewidth=2, label='Vertical Rudder (rad)'
        )
        self.rudder_h_line, = self.ax_control.plot(
            [], [], 'b-', linewidth=2, label='Horizontal Rudder (rad)'
        )
        self.ax_control.set_title('Control Inputs')
        self.ax_control.set_xlabel('Time (s)')
        self.ax_control.set_ylabel('Value')
        self.ax_control.legend(loc='best')
        self.ax_control.grid(True, alpha=0.3)

    def _init_error_plot(self):
        """Initialize tracking errors plot"""
        self.path_error_line, = self.ax_error.plot(
            [], [], 'r-', linewidth=2, label='Path Error (m)'
        )
        self.pitch_error_line, = self.ax_error.plot(
            [], [], 'g-', linewidth=2, label='Pitch Error (rad)'
        )
        self.yaw_error_line, = self.ax_error.plot(
            [], [], 'b-', linewidth=2, label='Yaw Error (rad)'
        )
        self.ax_error.set_title('Tracking Errors')
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Error')
        self.ax_error.legend(loc='best')
        self.ax_error.grid(True, alpha=0.3)

    def _init_obstacle_plot(self):
        """Initialize obstacle distance plot (if applicable)"""
        if self.has_obstacle:
            self.obstacle_distance_line, = self.ax_obstacle.plot(
                [], [], 'b-', linewidth=2, label='Distance to Obstacle (m)'
            )
            
            # Add reference lines for obstacle radius and safety distance
            if hasattr(self.env.config, '__getitem__'):
                if 'r_obstacle' in self.env.config:
                    self.ax_obstacle.axhline(
                        y=self.env.config['r_obstacle'],
                        color='r', linestyle='--', linewidth=1.5,
                        label='Collision Boundary', alpha=0.7
                    )
                if 'avoid_dis' in self.env.config and 'r_obstacle' in self.env.config:
                    safety_dist = self.env.config['r_obstacle'] + self.env.config['avoid_dis']
                    self.ax_obstacle.axhline(
                        y=safety_dist,
                        color='orange', linestyle='--', linewidth=1.5,
                        label='Safety Boundary', alpha=0.7
                    )
            
            self.ax_obstacle.set_title('Obstacle Distance')
            self.ax_obstacle.set_xlabel('Time (s)')
            self.ax_obstacle.set_ylabel('Distance (m)')
            self.ax_obstacle.legend(loc='best')
            self.ax_obstacle.grid(True, alpha=0.3)

    def _update_3d_plot(self):
        """Update 3D trajectory plot"""
        history_array = np.array(self.history_positions)
        
        # Update AUV position and trajectory
        self.auv_position._offsets3d = (
            [history_array[-1, 0]],
            [history_array[-1, 1]],
            [history_array[-1, 2]]
        )
        self.trajectory_line.set_data_3d(
            history_array[:, 0],
            history_array[:, 1],
            history_array[:, 2]
        )
        
        # Auto-adjust view limits
        margin = 5
        self.ax_3d.set_xlim([
            min(min(self.env.x_traj), np.min(history_array[:, 0])) - margin,
            max(max(self.env.x_traj), np.max(history_array[:, 0])) + margin
        ])
        self.ax_3d.set_ylim([
            min(min(self.env.y_traj), np.min(history_array[:, 1])) - margin,
            max(max(self.env.y_traj), np.max(history_array[:, 1])) + margin
        ])
        self.ax_3d.set_zlim([
            min(min(self.env.z_traj), np.min(history_array[:, 2])) - margin,
            max(max(self.env.z_traj), np.max(history_array[:, 2])) + margin
        ])
        
        # Update obstacle appearance based on collision
        if self.has_obstacle:
            current_pos = np.array(self.history_positions[-1])
            obstacle = self.env.obstacle
            
            # Check collision status
            collision_flag = obstacle.check_collision(current_pos, safety_margin=0.0)
            
            # Update obstacle color based on collision
            if collision_flag and not self.collision_occurred:
                self.collision_occurred = True
                # Change to collision color
                if self.obstacle_surface is not None:
                    self.obstacle_surface.set_facecolor('darkred')
                    self.obstacle_surface.set_alpha(0.8)
            
            # Calculate distance for info display
            distance_to_obstacle = obstacle.get_distance(current_pos)
            
            # Update info text
            info_text = f'Step: {self.env._count}'
            if self.has_obstacle:
                info_text += f'\nDistance: {distance_to_obstacle:.2f} m'
                info_text += f'\nCollision: {"Yes" if self.collision_occurred else "No"}'
            
            self.step_text.set_text(info_text)
        else:
            # No obstacle, just update step count
            self.step_text.set_text(f'Step: {self.env._count}')

    def _update_data_plots(self, render_config: Dict):
        """
        Update data curve plots
        
        Args:
            render_config: Rendering configuration dictionary
        """
        time_data = self.history_data['time']
        
        if render_config['show_control']:
            self.thrust_line.set_data(time_data, self.history_data['thrust'])
            self.rudder_v_line.set_data(time_data, self.history_data['rudder_v'])
            self.rudder_h_line.set_data(time_data, self.history_data['rudder_h'])
            self.ax_control.relim()
            self.ax_control.autoscale_view()
        
        if render_config['show_error']:
            self.path_error_line.set_data(time_data, self.history_data['path_error'])
            self.pitch_error_line.set_data(time_data, self.history_data['pitch_error'])
            self.yaw_error_line.set_data(time_data, self.history_data['yaw_error'])
            self.ax_error.relim()
            self.ax_error.autoscale_view()
        
        if self.has_obstacle and render_config.get('show_obstacle_info', False):
            if 'obstacle_distance' in self.history_data:
                self.obstacle_distance_line.set_data(time_data, self.history_data['obstacle_distance'])
                self.ax_obstacle.relim()
                self.ax_obstacle.autoscale_view()

    def render(self, mode: Optional[str] = None, **kwargs) -> Optional[np.ndarray]:
        """
        Update and display visualization
        
        Args:
            mode: Rendering mode (if None, uses environment's render_mode)
                  'human': Display window
                  'rgb_array': Return RGB image array
            **kwargs: Additional rendering configuration parameters
            
        Returns:
            RGB image array if mode is 'rgb_array', None otherwise
        """
        mode = mode or getattr(self.env, 'render_mode', 'human')
        
        # Update rendering configuration
        render_config = self.render_config.copy()
        render_config.update(kwargs)
        
        # Disable obstacle info if no obstacle present
        if not self.has_obstacle:
            render_config['show_obstacle_info'] = False
        
        # 记录数据
        self._record_data(render_config)
        
        # 初始化或更新图形
        self._setup_figure(render_config)
        
        # 更新图形
        self._update_plots(render_config)
        
        # 绘制但不显示
        self.fig.canvas.draw()
        
        if mode == 'rgb_array':
            # 将图形转换为RGB数组
            canvas = self.fig.canvas
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(int(height), int(width), 3)
            return image
        else:
            plt.draw()
            plt.pause(0.01)
            return None

    def _record_data(self, render_config: Dict):
        """
        Record history data
        
        Args:
            render_config: Rendering configuration dictionary
        """
        current_pos = np.array(self.env.x[:3].cpu() if hasattr(self.env.x, 'cpu') else self.env.x[:3])
        self.history_positions.append(current_pos.flatten())
        
        if render_config['show_control'] or render_config['show_error'] or \
           (self.has_obstacle and render_config.get('show_obstacle_info', False)):
            if self.history_data is None:
                self.history_data = {
                    'thrust': [],
                    'rudder_v': [],
                    'rudder_h': [],
                    'path_error': [],
                    'pitch_error': [],
                    'yaw_error': [],
                    'time': [],
                    'obstacle_distance': [] if self.has_obstacle else None
                }
            
            # Update history data
            self._update_history_data()

    def _update_history_data(self):
        """Update history data with current state"""
        self.history_data['thrust'].append(self._to_float(self.env.f[0]))
        self.history_data['rudder_v'].append(self._to_float(self.env.delta[0]))
        self.history_data['rudder_h'].append(self._to_float(self.env.delta[2]))
        self.history_data['path_error'].append(self._to_float(self.env.d_path))
        
        # Process angle errors
        desired_pitch = self._to_float(self.env.desired_states['pitch'])
        desired_yaw = self._to_float(self.env.desired_states['yaw'])
        current_pitch = self._to_float(self.env.x[4])
        current_yaw = self._to_float(self.env.x[5])
        
        pitch_error = self._limit_angle(desired_pitch - current_pitch)
        yaw_error = self._limit_angle(desired_yaw - current_yaw)
        
        self.history_data['pitch_error'].append(pitch_error)
        self.history_data['yaw_error'].append(yaw_error)
        self.history_data['time'].append(self.env._count * self.env.config['dt'])
        
        # Record obstacle distance if applicable
        if self.has_obstacle:
            current_pos = self.history_positions[-1]
            distance = self.env.obstacle.get_distance(current_pos)
            self.history_data['obstacle_distance'].append(distance)

    def _setup_figure(self, render_config: Dict):
        """
        Setup figure and subplots
        
        Args:
            render_config: Rendering configuration dictionary
        """
        if (self.fig is None or self._current_render_config != render_config):
            if self.fig is not None:
                plt.close(self.fig)
            
            self._init_figure(render_config)
            self._current_render_config = render_config.copy()

    def _init_figure(self, render_config: Dict):
        """
        Initialize figure with subplots
        
        Args:
            render_config: Rendering configuration dictionary
        """
        render_mode = getattr(self.env, 'render_mode', 'human')
        if render_mode == 'human':
            plt.ion()  # Interactive mode
        else:
            plt.ioff()  # Non-interactive mode for video generation
            
        self.fig = plt.figure(figsize=render_config['figure_size'], dpi=100)
        
        # Setup subplot layout
        self._setup_subplots(render_config)
        
        # Initialize plot objects
        self._init_plots(render_config)
        
        plt.tight_layout()

    def _setup_subplots(self, render_config: Dict):
        """
        Setup subplot layout
        
        Args:
            render_config: Rendering configuration dictionary
        """
        # Count number of subplots needed
        subplot_keys = ['show_control', 'show_error']
        if self.has_obstacle:
            subplot_keys.append('show_obstacle_info')
        
        num_data_plots = sum(1 for key in subplot_keys if render_config.get(key, False))
        
        if render_config['show_3d'] and num_data_plots > 0:
            # Layout: 3D plot on left, data plots on right
            gs = self.fig.add_gridspec(max(num_data_plots, 1), 2, width_ratios=[1.5, 1])
            self.ax_3d = self.fig.add_subplot(gs[:, 0], projection='3d')
            
            current_subplot = 0
            if render_config.get('show_control', False):
                self.ax_control = self.fig.add_subplot(gs[current_subplot, 1])
                current_subplot += 1
            
            if render_config.get('show_error', False):
                self.ax_error = self.fig.add_subplot(gs[current_subplot, 1])
                current_subplot += 1
            
            if self.has_obstacle and render_config.get('show_obstacle_info', False):
                self.ax_obstacle = self.fig.add_subplot(gs[current_subplot, 1])
        elif render_config['show_3d']:
            # Only 3D plot
            self.ax_3d = self.fig.add_subplot(111, projection='3d')
        else:
            # Only data plots (unusual case)
            current_subplot = 0
            if render_config.get('show_control', False):
                self.ax_control = self.fig.add_subplot(num_data_plots, 1, current_subplot + 1)
                current_subplot += 1
            if render_config.get('show_error', False):
                self.ax_error = self.fig.add_subplot(num_data_plots, 1, current_subplot + 1)
                current_subplot += 1
            if self.has_obstacle and render_config.get('show_obstacle_info', False):
                self.ax_obstacle = self.fig.add_subplot(num_data_plots, 1, current_subplot + 1)

    def _init_plots(self, render_config: Dict):
        """
        Initialize all plot objects
        
        Args:
            render_config: Rendering configuration dictionary
        """
        if render_config.get('show_3d', False):
            self._init_3d_plot()
        
        if render_config.get('show_control', False):
            self._init_control_plot()
        
        if render_config.get('show_error', False):
            self._init_error_plot()
        
        if self.has_obstacle and render_config.get('show_obstacle_info', False):
            self._init_obstacle_plot()

    def _update_plots(self, render_config: Dict):
        """
        Update all plots
        
        Args:
            render_config: Rendering configuration dictionary
        """
        if render_config.get('show_3d', False):
            self._update_3d_plot()
        
        if render_config.get('show_control', False) or render_config.get('show_error', False) or \
           (self.has_obstacle and render_config.get('show_obstacle_info', False)):
            self._update_data_plots(render_config)

    def close(self):
        """Close renderer and cleanup resources"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax_3d = None
            self.ax_control = None
            self.ax_error = None
            self.ax_obstacle = None
            self.history_positions = []
            self.history_data = None
            self.collision_occurred = False
            plt.ion()  # Restore interactive mode 