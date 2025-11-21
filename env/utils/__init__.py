"""
Utilities module for the AUV environment

This module provides utility classes for obstacle handling, trajectory generation,
and ocean current simulation.
"""

from .obstacle3d import Obstacle
from .AvoidObstacles import AvoidObstacles
from .ocean_current import OceanCurrent
from .RandomTraj3D import RandomTraj3D
from .trajectory_visualizer import (
    plot_3d_trajectory,
    plot_multiple_trajectories,
    create_evaluation_summary_plot
)

__all__ = [
    'Obstacle',
    'AvoidObstacles',
    'OceanCurrent',
    'RandomTraj3D',
    'plot_3d_trajectory',
    'plot_multiple_trajectories',
    'create_evaluation_summary_plot'
]

