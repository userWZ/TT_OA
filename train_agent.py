"""
Universal Training Script for Multiple RL Algorithms

Supports: PPO, SAC, TD3, DDPG, A2C, etc.
"""
import os
import numpy as np
import torch
from datetime import datetime
from typing import Callable, Dict, Any, Optional, Type

from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    EvalCallback, 
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.logger import configure

from env.auv_tracking_obstacle_avoidance_env import AUVTrackingObstacleAvoidanceEnv
from configs.auv_obstacle_avoidance_config import get_config


# Algorithm registry
ALGORITHMS = {
    'ppo': PPO,
    'sac': SAC,
    'td3': TD3,
    'ddpg': DDPG,
    'a2c': A2C,
}


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics"""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log custom metrics from environment
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    # Log terminal metrics
                    if 'terminal' in info:
                        terminal_info = info['terminal']
                        self.logger.record('rollout/ep_final_error', terminal_info.get('final_error', 0))
                        self.logger.record('rollout/ep_collision', int(terminal_info.get('collision_occurred', False)))
                        
                    # Log reward components
                    if 'reward_info' in info:
                        reward_info = info['reward_info']
                        self.logger.record('rollout/ep_path_error', reward_info.get('path_error', 0))
                        self.logger.record('rollout/ep_path_reward', reward_info.get('path_reward', 0))
                        self.logger.record('rollout/ep_pitch_reward', reward_info.get('pitch_reward', 0))
                        self.logger.record('rollout/ep_yaw_reward', reward_info.get('yaw_reward', 0))
                        
                        if 'obstacle_distance' in reward_info:
                            self.logger.record('rollout/ep_obstacle_distance', reward_info['obstacle_distance'])
                            self.logger.record('rollout/ep_obstacle_reward', reward_info.get('obstacle_reward', 0))
        
        return True


class ProgressBarCallback(BaseCallback):
    """Progress bar during training"""
    
    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None
        
    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.total_timesteps, desc="Training")
        except ImportError:
            self.pbar = None
            if self.verbose > 0:
                print("Install tqdm for progress bar: pip install tqdm")
    
    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True
    
    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


def create_env(config: dict, rank: int = 0, seed: int = 0) -> Callable:
    """Create a single environment instance"""
    def _init():
        env = AUVTrackingObstacleAvoidanceEnv(config)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_training_env(config: dict, n_envs: int = 4, seed: int = 0, use_subproc: bool = True):
    """Create vectorized training environment"""
    if use_subproc and n_envs > 1:
        env = SubprocVecEnv([create_env(config, i, seed) for i in range(n_envs)])
    else:
        env = make_vec_env(
            lambda: AUVTrackingObstacleAvoidanceEnv(config),
            n_envs=n_envs,
            seed=seed
        )
    
    env = VecMonitor(env)
    return env


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def get_algorithm_config(algorithm: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for each algorithm
    
    Args:
        algorithm: Algorithm name
        
    Returns:
        Dictionary of hyperparameters
    """
    configs = {
        'ppo': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=torch.nn.ReLU
            ),
        },
        'sac': {
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',
            'target_update_interval': 1,
            'target_entropy': 'auto',
            'use_sde': False,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            ),
        },
        'td3': {
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': (1, "episode"),
            'gradient_steps': -1,
            'policy_delay': 2,
            'target_policy_noise': 0.2,
            'target_noise_clip': 0.5,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            ),
        },
        'ddpg': {
            'learning_rate': 3e-4,
            'buffer_size': 1_000_000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': (1, "episode"),
            'gradient_steps': -1,
            'policy_kwargs': dict(
                net_arch=[256, 256],
                activation_fn=torch.nn.ReLU
            ),
        },
        'a2c': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_rms_prop': True,
            'normalize_advantage': False,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=torch.nn.ReLU
            ),
        },
    }
    
    return configs.get(algorithm.lower(), {})


def create_action_noise(algorithm: str, action_dim: int, noise_type: str = 'normal', noise_std: float = 0.1):
    """
    Create action noise for off-policy algorithms
    
    Args:
        algorithm: Algorithm name
        action_dim: Action space dimension
        noise_type: Type of noise ('normal' or 'ou')
        noise_std: Standard deviation of noise
        
    Returns:
        Action noise object or None
    """
    # Only off-policy algorithms use action noise
    if algorithm.lower() in ['td3', 'ddpg']:
        if noise_type == 'normal':
            return NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=noise_std * np.ones(action_dim)
            )
        elif noise_type == 'ou':
            return OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(action_dim),
                sigma=noise_std * np.ones(action_dim)
            )
    return None


def train(
    algorithm: str = 'ppo',
    config_name: str = 'with_obstacle',
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    use_normalize: bool = True,
    use_linear_schedule: bool = True,
    eval_freq: int = 10000,
    n_eval_episodes: int = 5,
    save_freq: int = 50000,
    seed: int = 0,
    device: str = 'auto',
    log_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    tensorboard_log: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    verbose: int = 1
) -> tuple:
    """
    Train RL agent with specified algorithm
    
    Args:
        algorithm: Algorithm name ('ppo', 'sac', 'td3', 'ddpg', 'a2c')
        config_name: Environment configuration
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        use_normalize: Whether to normalize observations
        use_linear_schedule: Whether to use linear LR schedule
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        save_freq: Model save frequency
        seed: Random seed
        device: Device to use
        log_dir: Log directory
        model_dir: Model save directory
        tensorboard_log: Tensorboard directory
        hyperparams: Custom hyperparameters (override defaults)
        verbose: Verbosity level
        
    Returns:
        Tuple of (model, model_dir)
    """
    # Validate algorithm
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ALGORITHMS.keys())}")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm.upper()}_{config_name}_{timestamp}"
    
    if log_dir is None:
        log_dir = f"logs/{run_name}"
    if model_dir is None:
        model_dir = f"models/{run_name}"
    if tensorboard_log is None:
        tensorboard_log = f"tensorboard/{run_name}"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    print("=" * 80)
    print(f"Training Configuration: {run_name}")
    print("=" * 80)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Environment: {config_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Model directory: {model_dir}")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load environment configuration
    env_config = get_config(config_name)
    
    # Get default hyperparameters
    default_hyperparams = get_algorithm_config(algorithm)
    
    # Override with custom hyperparameters
    if hyperparams:
        default_hyperparams.update(hyperparams)
    
    # Apply linear schedule if requested
    if use_linear_schedule and 'learning_rate' in default_hyperparams:
        default_hyperparams['learning_rate'] = linear_schedule(default_hyperparams['learning_rate'])
        if 'clip_range' in default_hyperparams:
            default_hyperparams['clip_range'] = linear_schedule(default_hyperparams['clip_range'])
    
    print(f"\nHyperparameters:")
    for key, value in default_hyperparams.items():
        if key != 'policy_kwargs':
            print(f"  {key}: {value}")
    
    # Create training environment
    print(f"\nCreating training environment...")
    
    # Adjust n_envs for off-policy algorithms
    if algorithm in ['sac', 'td3', 'ddpg'] and n_envs > 1:
        print(f"  Note: {algorithm.upper()} typically uses single environment, but {n_envs} will be used.")
    
    train_env = make_training_env(env_config, n_envs=n_envs, seed=seed)
    
    # Normalize observations if requested
    if use_normalize:
        print("  Wrapping with VecNormalize...")
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=default_hyperparams.get('gamma', 0.99)
        )
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_training_env(env_config, n_envs=1, seed=seed + 1000)
    if use_normalize:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=default_hyperparams.get('gamma', 0.99),
            training=False
        )
    
    # Create action noise for off-policy algorithms
    action_noise = None
    if algorithm in ['td3', 'ddpg']:
        action_dim = train_env.action_space.shape[0]
        action_noise = create_action_noise(algorithm, action_dim, noise_type='normal', noise_std=0.1)
        if action_noise is not None:
            print(f"  Using action noise: {type(action_noise).__name__}")
    
    # Create model
    print(f"\nInitializing {algorithm.upper()} model...")
    AlgorithmClass = ALGORITHMS[algorithm]
    
    model_kwargs = {
        'policy': "MlpPolicy",
        'env': train_env,
        'tensorboard_log': tensorboard_log,
        'verbose': verbose,
        'seed': seed,
        'device': device,
        **default_hyperparams
    }
    
    # Add action noise for off-policy algorithms
    if action_noise is not None:
        model_kwargs['action_noise'] = action_noise
    
    model = AlgorithmClass(**model_kwargs)
    
    # Setup logger
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    
    # Setup callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=model_dir,
        name_prefix=f"checkpoint_{algorithm}",
        save_replay_buffer=algorithm in ['sac', 'td3', 'ddpg'],
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # Tensorboard callback
    tensorboard_callback = TensorboardCallback(verbose=0)
    callbacks.append(tensorboard_callback)
    
    # Progress bar callback
    progress_callback = ProgressBarCallback(total_timesteps, verbose=1)
    callbacks.append(progress_callback)
    
    callback = CallbackList(callbacks)
    
    # Train the model
    print("\nStarting training...")
    print("=" * 80)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            tb_log_name=algorithm.upper(),
            reset_num_timesteps=True,
            progress_bar=False
        )
        
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user!")
        print("=" * 80)
    
    # Save final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save replay buffer for off-policy algorithms
    if algorithm in ['sac', 'td3', 'ddpg'] and hasattr(model, 'replay_buffer'):
        buffer_path = os.path.join(model_dir, "replay_buffer.pkl")
        model.save_replay_buffer(buffer_path)
        print(f"Replay buffer saved to: {buffer_path}")
    
    # Save VecNormalize statistics
    if use_normalize:
        norm_path = os.path.join(model_dir, "vecnormalize.pkl")
        train_env.save(norm_path)
        print(f"VecNormalize stats saved to: {norm_path}")
    
    # Close environments
    train_env.close()
    eval_env.close()
    
    print(f"\nAll artifacts saved to: {model_dir}")
    print(f"Tensorboard logs: {tensorboard_log}")
    print("\nView training progress:")
    print(f"  tensorboard --logdir {tensorboard_log}")
    
    return model, model_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent for AUV obstacle avoidance")
    
    # Algorithm selection
    parser.add_argument("--algo", type=str, default="ppo",
                       choices=list(ALGORITHMS.keys()),
                       help="RL algorithm to use")
    
    # Environment arguments
    parser.add_argument("--config", type=str, default="with_obstacle",
                       choices=['no_obstacle', 'with_obstacle', 'with_current', 'training', 'hard'],
                       help="Environment configuration")
    
    # Training arguments
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=5,
                       help="Number of evaluation episodes")
    parser.add_argument("--save-freq", type=int, default=50000,
                       help="Model save frequency")
    
    # Hyperparameters (optional overrides)
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (override default)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (override default)")
    parser.add_argument("--gamma", type=float, default=None,
                       help="Discount factor (override default)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       choices=['auto', 'cpu', 'cuda'],
                       help="Device to use")
    parser.add_argument("--no-normalize", action="store_true",
                       help="Disable observation normalization")
    parser.add_argument("--no-linear-schedule", action="store_true",
                       help="Disable linear learning rate schedule")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    # Prepare custom hyperparameters
    custom_hyperparams = {}
    if args.lr is not None:
        custom_hyperparams['learning_rate'] = args.lr
    if args.batch_size is not None:
        custom_hyperparams['batch_size'] = args.batch_size
    if args.gamma is not None:
        custom_hyperparams['gamma'] = args.gamma
    
    # Train model
    train(
        algorithm=args.algo,
        config_name=args.config,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        use_normalize=not args.no_normalize,
        use_linear_schedule=not args.no_linear_schedule,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_freq=args.save_freq,
        seed=args.seed,
        device=args.device,
        hyperparams=custom_hyperparams if custom_hyperparams else None,
        verbose=args.verbose
    )

