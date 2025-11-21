"""
Universal Agent Evaluation Script

Evaluate trained RL agents (PPO, SAC, TD3, DDPG, A2C, etc.)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import json

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config
from env.utils.trajectory_visualizer import create_evaluation_summary_plot


# Algorithm registry
ALGORITHMS = {
    'ppo': PPO,
    'sac': SAC,
    'td3': TD3,
    'ddpg': DDPG,
    'a2c': A2C,
}


def load_model(model_path: str, algorithm: str, env, verbose: bool = True):
    """
    Load trained model
    
    Args:
        model_path: Path to saved model
        algorithm: Algorithm name
        env: Environment
        verbose: Print information
        
    Returns:
        Loaded model
    """
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    AlgorithmClass = ALGORITHMS[algorithm]
    
    if verbose:
        print(f"Loading {algorithm.upper()} model from: {model_path}")
    
    model = AlgorithmClass.load(model_path, env=env)
    return model


def evaluate_agent(
    model_path: str,
    algorithm: str,
    config_name: str = 'with_obstacle',
    n_episodes: int = 10,
    normalize_path: Optional[str] = None,
    render: bool = False,
    save_plots: bool = True,
    save_trajectories: bool = True,
    output_dir: str = "evaluation_results",
    seed: int = 0,
    deterministic: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate trained agent
    
    Args:
        model_path: Path to saved model
        algorithm: Algorithm name
        config_name: Environment configuration
        n_episodes: Number of evaluation episodes
        normalize_path: Path to VecNormalize stats
        render: Whether to render episodes
        save_plots: Whether to save plots
        save_trajectories: Whether to save trajectory data
        output_dir: Output directory
        seed: Random seed
        deterministic: Use deterministic actions
        verbose: Print information
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("=" * 80)
    print(f"{algorithm.upper()} Agent Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Config: {config_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print("=" * 80 + "\n")
    
    # Create environment
    env_config = get_config(config_name)
    env = AUVTrackingObstacleAvoidanceEnv(env_config)
    env = DummyVecEnv([lambda: env])
    
    # Load VecNormalize if available
    if normalize_path and os.path.exists(normalize_path):
        if verbose:
            print(f"Loading VecNormalize stats from: {normalize_path}")
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load model
    model = load_model(model_path, algorithm, env, verbose)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_path_errors = []
    episode_collisions = []
    final_errors = []
    min_obstacle_distances = []
    
    # Trajectory data
    trajectories = []
    reference_trajectories = []
    obstacles_data = []
    
    # Run evaluation
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        collision_occurred = False
        min_obs_dist = float('inf')
        
        positions = []
        
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            episode_length += 1
            
            # Store position
            if hasattr(env.envs[0], 'x'):
                pos = env.envs[0].x[:3].flatten()
                positions.append(pos.copy())
            
            # Track metrics
            if 'reward_info' in info[0]:
                reward_info = info[0]['reward_info']
                if 'collision_occurred' in reward_info:
                    collision_occurred = reward_info['collision_occurred']
                if 'obstacle_distance' in reward_info:
                    min_obs_dist = min(min_obs_dist, reward_info['obstacle_distance'])
            
            if render:
                env.render()
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if 'terminal' in info[0]:
            final_errors.append(info[0]['terminal'].get('final_error', 0))
        
        episode_collisions.append(int(collision_occurred))
        
        if min_obs_dist != float('inf'):
            min_obstacle_distances.append(min_obs_dist)
        
        if len(positions) > 0:
            trajectories.append(np.array(positions))
            
            # Store reference trajectory (always append, even if None)
            if hasattr(env.envs[0], 'ref_traj') and env.envs[0].ref_traj is not None:
                reference_trajectories.append(env.envs[0].ref_traj.copy())
            else:
                reference_trajectories.append(None)
            
            # Store obstacle information (always append, even if None)
            if hasattr(env.envs[0], 'obstacle') and env.envs[0].obstacle is not None:
                obstacle = env.envs[0].obstacle
                # Get safety distance from config
                safe_dist = env_config.get('obstacle_avoidance', {}).get('safe_distance', 
                            env_config.get('avoid_dis', 0.0))
                obstacles_data.append({
                    'position': obstacle.position.copy(),
                    'radius': obstacle.radius,
                    'safety_distance': safe_dist,
                    'collision_occurred': collision_occurred
                })
            else:
                obstacles_data.append(None)
        
        if verbose:
            # Format obstacle distance safely
            if min_obs_dist != float('inf'):
                dist_str = f"{min_obs_dist:.3f}"
            else:
                dist_str = "N/A"
            
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Length={episode_length}, "
                  f"Error={final_errors[-1] if final_errors else 0:.3f}, "
                  f"MinDist={dist_str}, "
                  f"Collision={'Yes' if collision_occurred else 'No'}")
    
    # Calculate statistics
    results = {
        'algorithm': algorithm.upper(),
        'config': config_name,
        'n_episodes': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_errors': final_errors,
        'collision_rate': np.mean(episode_collisions),
        'min_obstacle_distances': min_obstacle_distances,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'mean_final_error': np.mean(final_errors) if final_errors else 0,
        'std_final_error': np.std(final_errors) if final_errors else 0,
        'mean_min_obstacle_dist': np.mean(min_obstacle_distances) if min_obstacle_distances else 0,
    }
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Episodes: {n_episodes}")
    print(f"\nReward:")
    print(f"  Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean: {results['mean_length']:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"\nFinal Path Error:")
    print(f"  Mean: {results['mean_final_error']:.3f} ± {results['std_final_error']:.3f}")
    if final_errors:
        print(f"  Min: {np.min(final_errors):.3f}")
        print(f"  Max: {np.max(final_errors):.3f}")
    if min_obstacle_distances:
        print(f"\nMin Obstacle Distance:")
        print(f"  Mean: {results['mean_min_obstacle_dist']:.3f}")
    print(f"\nCollision Rate:")
    print(f"  {results['collision_rate']*100:.1f}% ({np.sum(episode_collisions)}/{n_episodes})")
    print("=" * 80)
    
    # Save results and plots
    if save_plots or save_trajectories:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        results_path = os.path.join(output_dir, f'{algorithm}_evaluation_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types
            json_results = {k: (v.tolist() if isinstance(v, np.ndarray) else 
                               float(v) if isinstance(v, (np.float32, np.float64)) else v)
                           for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    if save_plots and len(trajectories) > 0:
        # Debug information
        if verbose:
            print(f"\nPlotting {len(trajectories)} trajectories")
            print(f"  - Reference trajectories: {sum(1 for r in reference_trajectories if r is not None)}/{len(reference_trajectories)}")
            print(f"  - Obstacles: {sum(1 for o in obstacles_data if o is not None)}/{len(obstacles_data)}")
            if obstacles_data and obstacles_data[0]:
                print(f"  - Sample obstacle: pos={obstacles_data[0]['position']}, "
                      f"radius={obstacles_data[0]['radius']}, "
                      f"safety={obstacles_data[0]['safety_distance']}")
        
        # Create comprehensive evaluation plots using the new visualizer
        create_evaluation_summary_plot(
            trajectories=trajectories,
            reference_trajectories=reference_trajectories,
            obstacle_data=obstacles_data,
            metrics=results,
            algorithm=algorithm,
            output_dir=output_dir
        )
    
    env.close()
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("--algo", type=str, required=True,
                       choices=list(ALGORITHMS.keys()),
                       help="Algorithm used")
    parser.add_argument("--config", type=str, default="with_obstacle",
                       help="Environment configuration")
    parser.add_argument("--n-episodes", type=int, default=1,
                       help="Number of evaluation episodes")
    parser.add_argument("--normalize", type=str, default=None,
                       help="Path to VecNormalize stats")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--no-plots", action="store_true",
                       help="Don't save plots")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic actions")
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model_path,
        algorithm=args.algo,
        config_name=args.config,
        n_episodes=args.n_episodes,
        normalize_path=args.normalize,
        render=args.render,
        save_plots=not args.no_plots,
        output_dir=args.output_dir,
        seed=args.seed,
        deterministic=not args.stochastic
    )

