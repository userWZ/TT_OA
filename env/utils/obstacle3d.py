"""
3D Spherical Obstacle Module

Provides spherical obstacle representation with collision detection
and distance calculation capabilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Obstacle:
    """
    3D Spherical Obstacle
    
    Represents a spherical obstacle in 3D space with collision detection
    and distance calculation capabilities.
    
    Attributes:
        position (np.ndarray): Obstacle center position [x, y, z]
        radius (float): Obstacle radius in meters
        observed (bool): Whether obstacle has been observed
        collided (bool): Whether collision has occurred
    """
    
    def __init__(self, radius: float, position: list):
        """
        Initialize obstacle
        
        Args:
            radius: Obstacle radius in meters
            position: Center position [x, y, z]
        """
        self.position = np.array(position)
        self.radius = radius
        self.observed = False
        self.collided = False

    def return_plot_variables(self):
        """
        Generate mesh coordinates for plotting obstacle surface
        
        Returns:
            List of [x, y, z] coordinate arrays for surface plotting
        """
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.position[0] + self.radius*np.cos(u)*np.sin(v)
        y = self.position[1] + self.radius*np.sin(u)*np.sin(v)
        z = self.position[2] + self.radius*np.cos(v)
        return [x, y, z]
    
    def check_collision(self, position, safety_margin: float = 0.0) -> bool:
        """
        Check if given position collides with obstacle
        
        Args:
            position: Position to check [x, y, z] or np.ndarray
            safety_margin: Additional safety distance in meters (default: 0.0)
            
        Returns:
            True if collision detected, False otherwise
        """
        position = np.array(position).flatten()[:3]  # Ensure it's a 1D array with 3 elements
        distance = np.linalg.norm(position - self.position)
        return distance < (self.radius + safety_margin)
    
    def get_distance(self, position) -> float:
        """
        Calculate distance from position to obstacle surface
        
        Args:
            position: Position to check [x, y, z] or np.ndarray
            
        Returns:
            Distance to obstacle surface in meters
            (negative if inside obstacle)
        """
        position = np.array(position).flatten()[:3]  # Ensure it's a 1D array with 3 elements
        distance_to_center = np.linalg.norm(position - self.position)
        return distance_to_center - self.radius


if __name__ == "__main__":
    # Example usage and visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create obstacles
    o1 = Obstacle(3, [0, 0, 0])
    ax.plot_surface(*o1.return_plot_variables(), alpha=0.6, color='blue')
    
    o2 = Obstacle(5, [10, 0, 0])
    ax.plot_surface(*o2.return_plot_variables(), alpha=0.6, color='red')
    
    # Test collision detection
    test_pos = np.array([2, 0, 0])
    print(f"Position {test_pos}:")
    print(f"  Collision with o1: {o1.check_collision(test_pos)}")
    print(f"  Distance to o1: {o1.get_distance(test_pos):.2f} m")
    print(f"  Collision with o2: {o2.check_collision(test_pos)}")
    print(f"  Distance to o2: {o2.get_distance(test_pos):.2f} m")
    
    # Plot test position
    ax.scatter(*test_pos, c='green', s=100, marker='o', label='Test Position')
    
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    plt.show()

