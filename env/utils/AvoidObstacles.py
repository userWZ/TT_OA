"""
Obstacle avoidance trajectory generation module

This module provides functionality for generating trajectories that avoid
spherical obstacles using perpendicular circle paths and cubic spline interpolation.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
from typing import Tuple, List, Optional
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from env.utils.obstacle3d import Obstacle


class AvoidObstacles:
    """
    Obstacle avoidance trajectory generator
    
    Generates smooth avoidance trajectories around spherical obstacles using
    perpendicular circle paths and cubic spline interpolation.
    
    Attributes:
        r_obstacle (float): Radius of the obstacle
        avoid_dis (float): Safety distance from obstacle surface
        clip_parts (int): Number of segments to divide the avoidance arc
    """
    
    def __init__(self, r_obstacle: float, avoid_dis: float, clip_parts: int = 5):
        """
        Initialize obstacle avoidance trajectory generator
        
        Args:
            r_obstacle: Obstacle radius in meters
            avoid_dis: Safety distance from obstacle surface in meters
            clip_parts: Number of segments for arc division (default: 5)
        """
        self.r_obstacle = r_obstacle
        self.avoid_dis = avoid_dis
        self.clip_parts = clip_parts

    def angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate angle between two vectors
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle between vectors in radians
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return theta

    def find_perpendicular_circle(
        self, 
        center: List[float], 
        radius: float, 
        point1: List[float], 
        point2: List[float], 
        time_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find perpendicular circle path on sphere surface between two points
        
        Given a sphere center, radius, and two points on the sphere surface,
        compute a circular arc on the sphere that connects the two points.
        
        Args:
            center: Sphere center coordinates [x, y, z]
            radius: Sphere radius
            point1: First point on sphere surface [x, y, z]
            point2: Second point on sphere surface [x, y, z]
            time_steps: Number of points on the arc
            
        Returns:
            Tuple of (x, y, z) coordinates arrays along the arc
        """
        # Calculate vectors and normal vector for the circle
        vector_1 = [x - y for x, y in zip(point1, center)]  # Vector OA
        vector_2 = [x - y for x, y in zip(point2, center)]  # Vector OB
        vector_3 = [x - y for x, y in zip(point1, point2)]  # Vector AB
        normal_vector = np.cross(vector_1, vector_3)        # Normal vector of the tangent circle
        normal_vector /= np.linalg.norm(normal_vector)      # Normalize

        # Calculate arc angle (smaller arc path)
        avoid_angle = self.angle_between_vectors(vector_1, vector_2)
        
        # Find the starting point of the arc (using angle as parameter)
        vector_4 = [radius, 0, (-normal_vector[0] * radius / normal_vector[2])]
        start_angle = self.angle_between_vectors(vector_1, vector_4)
        
        # Determine the positional relationship between v4 and OA
        x_temp = radius * np.cos(start_angle) + center[0]
        y_temp = radius * np.sin(start_angle) + center[1]
        z_temp = -normal_vector[0] * x_temp - normal_vector[1] * y_temp
        z_temp /= normal_vector[2]
        z_temp += center[2]

        # Adjust angle direction based on point proximity
        if (point1[0] - x_temp)**2 < 1 and (point1[1] - y_temp)**2 < 1 and (point1[2] - z_temp)**2 < 1:
            pass  # Keep start_angle as is
        else:
            start_angle = -start_angle

        # Generate arc points
        end_angle = start_angle + avoid_angle
        t_avoid = np.linspace(start_angle, end_angle, time_steps)

        # Calculate points on the arc
        x = radius * np.cos(t_avoid)
        y = radius * np.sin(t_avoid)
        z = -normal_vector[0] * x - normal_vector[1] * y
        z /= normal_vector[2]

        # Translate arc center to sphere center
        x += center[0]
        y += center[1]
        z += center[2]

        return x, y, z

    def cubic_spline_3d(
        self, 
        nums: int, 
        cubic_points: List[np.ndarray], 
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform 3D cubic spline interpolation
        
        Interpolates a smooth 3D curve through a set of control points using
        cubic spline interpolation.
        
        Args:
            nums: Number of control points
            cubic_points: List of control points, each as [x, y, z]
            num_points: Number of interpolated points to generate
            
        Returns:
            Tuple of (x_new, y_new, z_new) interpolated coordinate arrays
        """
        # Extract coordinates from control points
        x = []
        y = []
        z = []
        for i in range(nums):
            x.append(cubic_points[i][0])
            y.append(cubic_points[i][1])
            z.append(cubic_points[i][2])

        # Create parameter array for interpolation
        t = np.linspace(0, nums - 1, nums)
        
        # Create cubic spline interpolators for each dimension
        f_x = interp1d(t, x, kind='cubic')
        f_y = interp1d(t, y, kind='cubic')
        f_z = interp1d(t, z, kind='cubic')

        # Generate dense parameter array
        t_new = np.linspace(t[0], t[-1], num_points)

        # Perform interpolation
        x_new = f_x(t_new)
        y_new = f_y(t_new)
        z_new = f_z(t_new)

        return x_new, y_new, z_new

    def _intersection_equation(
        self, 
        t: np.ndarray, 
        center_x: float, 
        center_y: float, 
        center_z: float, 
        r_avoid: float
    ) -> np.ndarray:
        """
        Equation for finding trajectory-sphere intersection
        
        Computes the equation: (x-x0)^2 + (y-y0)^2 + (z-z0)^2 = R^2
        for finding time points where the trajectory intersects with the
        avoidance sphere.
        
        Args:
            t: Time parameter(s)
            center_x: Obstacle center x-coordinate
            center_y: Obstacle center y-coordinate
            center_z: Obstacle center z-coordinate
            r_avoid: Avoidance radius (obstacle radius + safety distance)
            
        Returns:
            Equation residual values
        """
        x, y, z = self.trajectory(t)
        a = (x - center_x)**2
        b = (y - center_y)**2
        c = (z - center_z)**2
        eq = a + b + c - r_avoid**2
        return np.squeeze([eq])

    def _solve_intersections(
        self, 
        t_left: float, 
        t_right: float, 
        center_x: float, 
        center_y: float, 
        center_z: float, 
        r_avoid: float
    ) -> Tuple[List[float], bool]:
        """
        Solve for trajectory-obstacle intersection points
        
        Finds time parameters where the reference trajectory intersects with
        the obstacle avoidance sphere.
        
        Args:
            t_left: Left bound of time interval to search
            t_right: Right bound of time interval to search
            center_x: Obstacle center x-coordinate
            center_y: Obstacle center y-coordinate
            center_z: Obstacle center z-coordinate
            r_avoid: Avoidance radius
            
        Returns:
            Tuple of (intersection_times, needs_avoidance)
            - intersection_times: List of time parameters at intersections
            - needs_avoidance: Boolean indicating if avoidance is needed
        """
        # Solve equation system with improved initial guesses
        t_initial_guess = np.linspace(t_left, t_right, 100)
        
        # Suppress fsolve warnings for cleaner output
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            t_solution = fsolve(
                self._intersection_equation, 
                t_initial_guess, 
                args=(center_x, center_y, center_z, r_avoid),
                full_output=False,
                xtol=1e-6,
                maxfev=1000
            )

        # Verify solutions are approximate solutions to the equation
        zeros = np.zeros(len(t_solution))
        result = np.isclose(
            self._intersection_equation(t_solution, center_x, center_y, center_z, r_avoid),
            zeros,
            rtol=3,
            atol=3
        )
        t_solution_temp = t_solution[result]

        # Extract unique positive solutions
        unique_num = set(round(num, 1) for num in t_solution_temp if num > 0)

        # Convert to sorted list
        t_inter = sorted([float(num) for num in unique_num])

        # Determine if avoidance is needed
        if len(t_inter) == 0 or len(t_inter) == 1 or (t_inter[-1] - t_inter[0]) < 3:
            needs_avoidance = False
        else:
            needs_avoidance = True
        
        return t_inter, needs_avoidance

    def trajectory(self, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate reference spiral trajectory
        
        Generates a 3D spiral trajectory that descends over time.
        
        Args:
            t: Time parameter(s)
            
        Returns:
            Tuple of (x, y, z) coordinates along the trajectory
        """
        x = (30 - 0.2 * t) * np.sin((math.pi / 25) * t)
        y = (30 - 0.2 * t) * np.cos((math.pi / 25) * t)
        z = 60 - 0.6 * t
        return x, y, z

    def s_curve_trajectory(
        self, 
        t: np.ndarray, 
        segments: int = 5, 
        amplitude: float = 2.0, 
        length: float = 20.0, 
        T: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate S-curve trajectory
        
        Generates a segmented sinusoidal trajectory in the horizontal plane.
        
        Args:
            t: Time parameter(s)
            segments: Number of sinusoidal segments
            amplitude: Amplitude of sinusoidal motion
            length: Total length of trajectory
            T: Total time duration
            
        Returns:
            Tuple of (x, y, z) coordinates along the trajectory
        """
        tau = t / T  # Normalize to [0, 1]
        x = length * tau

        seg_len = 1.0 / segments
        seg_idx = np.floor(tau / seg_len).astype(int)
        seg_idx = np.clip(seg_idx, 0, segments - 1)

        local_tau = (tau - seg_idx * seg_len) / seg_len

        phase = seg_idx * math.pi
        y = amplitude * np.sin(2 * math.pi * local_tau + phase)

        z = np.zeros_like(x)

        return x, y, z

    def avoid_obstacles_trajectory(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Obstacle,
        List, List, List, float, float, float, np.ndarray
    ]:
        """
        Generate trajectory with obstacle avoidance
        
        Creates a reference trajectory with automatic obstacle avoidance.
        A random obstacle is placed along the trajectory, and if intersection
        is detected, an avoidance path is generated using perpendicular circle
        and cubic spline interpolation.
        
        Returns:
            Tuple containing:
            - x: X-coordinates of trajectory with avoidance
            - y: Y-coordinates of trajectory with avoidance
            - z: Z-coordinates of trajectory with avoidance
            - obstacle: Obstacle object
            - x_inter: X-coordinates of intersection points
            - y_inter: Y-coordinates of intersection points
            - z_inter: Z-coordinates of intersection points
            - x_last: Last x-coordinate before trajectory start
            - y_last: Last y-coordinate before trajectory start
            - z_last: Last z-coordinate before trajectory start
            - avoid_info: Array containing obstacle information
        """
        # Generate reference trajectory
        t = np.arange(0, 100.2, 0.1)
        x, y, z = self.trajectory(t)
        
        # Get trajectory point before start (for reference)
        t_last = -1
        x_last, y_last, z_last = self.trajectory(t_last)

        # Generate random obstacle along trajectory
        random_time = random.uniform(4, 94)
        center_t = round(random_time, 1)
        center_x, center_y, center_z = self.trajectory(center_t)
        obstacle = Obstacle(self.r_obstacle, [center_x, center_y, center_z])
        r_avoid = self.r_obstacle + self.avoid_dis

        # Search for intersections in neighborhood of obstacle
        t_left = np.clip(center_t - 10, 0, 100)
        t_right = np.clip(center_t + 10, 0, 100)
 
        t_inter, needs_avoidance = self._solve_intersections(
            t_left, t_right, center_x, center_y, center_z, r_avoid
        )

        if needs_avoidance:
            # Convert time intersections to step indices
            step_inter_left = math.floor(t_inter[0] / 0.1)
            step_inter_right = math.ceil(t_inter[-1] / 0.1)
            t_inter_left = step_inter_left * 0.1
            t_inter_right = step_inter_right * 0.1
            
            # Calculate intersection point coordinates
            x_inter, y_inter, z_inter = self.trajectory(
                np.array([t_inter_left, t_inter_right])
            )
            p1_inter = [x_inter[0], y_inter[0], z_inter[0]]
            p2_inter = [x_inter[1], y_inter[1], z_inter[1]]

            avoid_steps = step_inter_right - step_inter_left

            # Generate perpendicular circle avoidance path
            x_avoid, y_avoid, z_avoid = self.find_perpendicular_circle(
                center=[center_x, center_y, center_z], 
                radius=r_avoid,
                point1=p1_inter,
                point2=p2_inter,
                time_steps=avoid_steps
            )

            # Divide arc into segments and find split points
            cubic_arr = np.array_split(
                np.array([x_avoid, y_avoid, z_avoid]),
                self.clip_parts,
                axis=1
            )
            split_points = [sub_arr[:, -1] for sub_arr in cubic_arr[:-1]]

            # Create control points for cubic spline with extended trajectory points
            cubic_points = split_points

            start_avoid_step = step_inter_left - 15
            end_avoid_step = step_inter_right + 20
            avoid_steps = end_avoid_step - start_avoid_step

            # Add trajectory points before and after avoidance region
            x0, y0, z0 = self.trajectory(start_avoid_step / 10)
            cubic_points[0] = np.array([x0, y0, z0])
            cubic_points.extend(
                [np.array(self.trajectory(end_avoid_step / 10))]
            )
            
            # Generate smooth avoidance trajectory using cubic spline
            x_avoid, y_avoid, z_avoid = self.cubic_spline_3d(
                nums=len(cubic_points),
                cubic_points=cubic_points,
                num_points=avoid_steps
            )
            
            # Replace original trajectory with avoidance trajectory
            for i in range(start_avoid_step + 1, end_avoid_step - 1):
                j = i - start_avoid_step
                x[i], y[i], z[i] = x_avoid[j], y_avoid[j], z_avoid[j]
        else:
            # No avoidance needed
            x_inter, y_inter, z_inter = [[0.0]], [[0.0]], [[0.0]]

        # Pack obstacle information
        avoid_info = np.array([
            center_x, center_y, center_z,
            self.r_obstacle, self.avoid_dis,
            x_inter, y_inter, z_inter
        ], dtype=object)

        return x, y, z, obstacle, x_inter, y_inter, z_inter, x_last, y_last, z_last, avoid_info
    
def visualize_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    obstacle: Obstacle,
    x_inter: List,
    y_inter: List,
    z_inter: List
) -> None:
    """
    Visualize avoidance trajectory with obstacle
    
    Args:
        x: X-coordinates of trajectory
        y: Y-coordinates of trajectory
        z: Z-coordinates of trajectory
        obstacle: Obstacle object
        x_inter: X-coordinates of intersection points
        y_inter: Y-coordinates of intersection points
        z_inter: Z-coordinates of intersection points
    """
    fig = plt.figure(figsize=(14, 6))
    
    # 3D view from angle
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=2, label='Avoidance Trajectory')
    ax1.plot_surface(*obstacle.return_plot_variables(), alpha=0.3, color='red')
    ax1.scatter(x_inter, y_inter, z_inter, marker='o', color='r', s=100,
                label='Intersection Points', edgecolors='black', linewidths=2)
    ax1.set_xlabel('X (East)')
    ax1.set_ylabel('Y (North)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('3D View')
    ax1.legend()
    ax1.invert_zaxis()

    # Top-down view
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x, y, z, 'b-', linewidth=2, label='Avoidance Trajectory')
    ax2.plot_surface(*obstacle.return_plot_variables(), alpha=0.3, color='red')
    ax2.scatter(x_inter, y_inter, z_inter, marker='o', color='r', s=100,
                label='Intersection Points', edgecolors='black', linewidths=2)
    ax2.set_xlabel('X (East)')
    ax2.set_ylabel('Y (North)')
    ax2.set_zlabel('Z (Depth)')
    ax2.set_title('Top View')
    ax2.view_init(elev=90, azim=-90)
    ax2.legend()
    ax2.invert_zaxis()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Generating trajectory with obstacle avoidance...")
    
    avoider = AvoidObstacles(r_obstacle=3.0, avoid_dis=3.0, clip_parts=5)
    x, y, z, obstacle, x_inter, y_inter, z_inter, _, _, _, _ = avoider.avoid_obstacles_trajectory()

    print(f"Trajectory length: {len(x)} points")
    print(f"Obstacle position: [{obstacle.position[0]:.2f}, {obstacle.position[1]:.2f}, {obstacle.position[2]:.2f}]")
    print(f"Obstacle radius: {obstacle.radius:.2f} m")
    
    # Visualize
    visualize_trajectory(x, y, z, obstacle, x_inter, y_inter, z_inter)

