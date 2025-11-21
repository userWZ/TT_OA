# 🚀 Training Guide for AUV Obstacle Avoidance

Complete guide for training and evaluating RL agents on AUV obstacle avoidance tasks.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Supported Algorithms](#supported-algorithms)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)

## 🎯 Quick Start

### Basic Training (PPO)

```bash
# Train PPO agent with default settings
python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000

# Train SAC agent
python train_agent.py --algo sac --config with_obstacle --timesteps 1000000

# Train with 8 parallel environments
python train_agent.py --algo ppo --config with_obstacle --n-envs 8
```

### Basic Evaluation

```bash
# Evaluate trained model
python evaluate_agent.py models/PPO_with_obstacle_20251120_120000/best_model/best_model.zip \
    --algo ppo --config with_obstacle --n-episodes 20

# Evaluate with rendering
python evaluate_agent.py models/SAC_with_obstacle_20251120_130000/best_model/best_model.zip \
    --algo sac --config with_obstacle --render
```

## 🤖 Supported Algorithms

| Algorithm | Type | Best For | Pros | Cons |
|-----------|------|----------|------|------|
| **PPO** | On-Policy | General tasks | Stable, sample efficient | Requires tuning |
| **SAC** | Off-Policy | Continuous control | Very stable, efficient | Slower per step |
| **TD3** | Off-Policy | High-dimensional | Robust, efficient | Needs tuning |
| **DDPG** | Off-Policy | Continuous control | Fast learning | Less stable |
| **A2C** | On-Policy | Quick prototyping | Simple, fast | Less efficient |

### Algorithm Comparison

```bash
# PPO - Most reliable for this task
python train_agent.py --algo ppo --config with_obstacle

# SAC - Better sample efficiency, slightly slower
python train_agent.py --algo sac --config with_obstacle

# TD3 - Good balance of speed and stability
python train_agent.py --algo td3 --config with_obstacle
```

## 🎓 Training

### Configuration Options

#### Environment Configurations

```bash
# No obstacle (basic trajectory tracking)
python train_agent.py --algo ppo --config no_obstacle

# With obstacle (main task)
python train_agent.py --algo ppo --config with_obstacle

# With obstacle + ocean current
python train_agent.py --algo ppo --config with_current

# Training configuration (uses fuzzy parameters)
python train_agent.py --algo ppo --config training

# Hard mode (obstacle + strong current + fuzzy)
python train_agent.py --algo ppo --config hard
```

#### Common Training Arguments

```bash
python train_agent.py \
    --algo ppo \                      # Algorithm: ppo, sac, td3, ddpg, a2c
    --config with_obstacle \          # Environment config
    --timesteps 2000000 \             # Total training steps
    --n-envs 8 \                      # Parallel environments
    --eval-freq 20000 \               # Evaluation frequency
    --n-eval-episodes 10 \            # Episodes per evaluation
    --save-freq 100000 \              # Model save frequency
    --seed 42 \                       # Random seed
    --device cuda \                   # Device: cpu, cuda, auto
    --verbose 1                       # Verbosity: 0, 1, 2
```

#### Hyperparameter Overrides

```bash
# Override learning rate
python train_agent.py --algo ppo --lr 1e-4

# Override batch size
python train_agent.py --algo sac --batch-size 512

# Override discount factor
python train_agent.py --algo td3 --gamma 0.995
```

### Advanced Training Options

#### Disable Observation Normalization

```bash
python train_agent.py --algo ppo --no-normalize
```

#### Disable Learning Rate Schedule

```bash
python train_agent.py --algo ppo --no-linear-schedule
```

#### GPU Training

```bash
# Automatic device selection
python train_agent.py --algo ppo --device auto

# Force GPU
python train_agent.py --algo ppo --device cuda

# Specific GPU
CUDA_VISIBLE_DEVICES=1 python train_agent.py --algo ppo --device cuda
```

### Training Examples

#### Example 1: Quick Training (PPO)

```bash
# Fast training for testing
python train_agent.py \
    --algo ppo \
    --config with_obstacle \
    --timesteps 500000 \
    --n-envs 4 \
    --eval-freq 10000 \
    --seed 0
```

#### Example 2: Full Training (SAC)

```bash
# Full training with SAC
python train_agent.py \
    --algo sac \
    --config training \
    --timesteps 2000000 \
    --n-envs 1 \
    --eval-freq 20000 \
    --n-eval-episodes 10 \
    --save-freq 50000 \
    --device cuda \
    --seed 42
```

#### Example 3: Hyperparameter Tuning (TD3)

```bash
# Custom hyperparameters
python train_agent.py \
    --algo td3 \
    --config with_obstacle \
    --timesteps 1000000 \
    --lr 1e-4 \
    --batch-size 512 \
    --gamma 0.995 \
    --device cuda
```

## 📊 Evaluation

### Basic Evaluation

```bash
# Evaluate best model
python evaluate_agent.py \
    models/PPO_with_obstacle_20251120/best_model/best_model.zip \
    --algo ppo \
    --config with_obstacle \
    --n-episodes 20
```

### Evaluation with VecNormalize

```bash
# If you used observation normalization during training
python evaluate_agent.py \
    models/PPO_with_obstacle_20251120/best_model/best_model.zip \
    --algo ppo \
    --config with_obstacle \
    --normalize models/PPO_with_obstacle_20251120/vecnormalize.pkl \
    --n-episodes 20
```

### Evaluation Options

```bash
python evaluate_agent.py MODEL_PATH \
    --algo ppo \                      # Algorithm used
    --config with_obstacle \          # Environment config
    --n-episodes 20 \                 # Number of episodes
    --normalize PATH \                # VecNormalize stats
    --render \                        # Show visualization
    --output-dir results \            # Output directory
    --seed 42 \                       # Random seed
    --stochastic                      # Use stochastic actions
```

### Evaluation Outputs

After evaluation, you'll find:

```
evaluation_results/
├── ppo_evaluation_results.json      # Numerical results
├── ppo_evaluation_metrics.png       # Metric plots
└── ppo_sample_trajectories.png      # 3D trajectory plots
```

## 🔧 Hyperparameter Tuning

### PPO Hyperparameters

```python
# Default PPO hyperparameters
{
    'learning_rate': 3e-4,           # Higher for faster learning
    'n_steps': 2048,                 # Rollout length
    'batch_size': 64,                # Minibatch size
    'n_epochs': 10,                  # Updates per rollout
    'gamma': 0.99,                   # Discount factor
    'gae_lambda': 0.95,              # GAE parameter
    'clip_range': 0.2,               # PPO clip range
    'ent_coef': 0.01,                # Entropy coefficient
    'vf_coef': 0.5,                  # Value function coef
    'max_grad_norm': 0.5,            # Gradient clipping
}
```

**Tuning Tips:**
- Increase `learning_rate` (e.g., 5e-4) for faster convergence
- Increase `n_steps` (e.g., 4096) for better sample efficiency
- Increase `batch_size` (e.g., 128) with more compute
- Adjust `ent_coef` (e.g., 0.001-0.1) for exploration

### SAC Hyperparameters

```python
# Default SAC hyperparameters
{
    'learning_rate': 3e-4,
    'buffer_size': 1_000_000,        # Replay buffer size
    'learning_starts': 10000,        # Random steps before learning
    'batch_size': 256,               # Larger than PPO
    'tau': 0.005,                    # Soft update coefficient
    'gamma': 0.99,
    'train_freq': 1,                 # Train every step
    'gradient_steps': 1,             # Gradient steps per step
    'ent_coef': 'auto',              # Automatic entropy tuning
}
```

**Tuning Tips:**
- Increase `buffer_size` (e.g., 2M) for better sample reuse
- Adjust `learning_starts` based on task complexity
- Set `train_freq` to (1, "episode") for episode-based training

### TD3 Hyperparameters

```python
# Default TD3 hyperparameters
{
    'learning_rate': 3e-4,
    'buffer_size': 1_000_000,
    'learning_starts': 10000,
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'policy_delay': 2,               # Update policy every N steps
    'target_policy_noise': 0.2,      # Target policy smoothing
    'target_noise_clip': 0.5,        # Noise clipping
}
```

## 📈 Monitoring

### TensorBoard

View training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir tensorboard/

# View specific run
tensorboard --logdir tensorboard/PPO_with_obstacle_20251120_120000/
```

**Key Metrics to Monitor:**

- `rollout/ep_rew_mean`: Average episode reward
- `rollout/ep_len_mean`: Average episode length
- `rollout/ep_final_error`: Final path tracking error
- `rollout/ep_collision`: Collision rate
- `rollout/ep_obstacle_distance`: Minimum obstacle distance
- `train/learning_rate`: Current learning rate
- `train/loss`: Training loss

### Training Progress

Monitor training in terminal:

```
Training Configuration: PPO_with_obstacle_20251120_120000
================================================================================
Algorithm: PPO
Environment: with_obstacle
Total timesteps: 1,000,000
Parallel environments: 4
...

Training:  15%|████▍                   | 150000/1000000 [05:23<30:44]
```

### Evaluation During Training

Check `logs/` directory for evaluation results:

```
logs/PPO_with_obstacle_20251120/
├── evaluations.npz                  # Evaluation metrics
├── progress.csv                     # Training progress
└── best_model/                      # Best model checkpoint
    └── best_model.zip
```

## 🎯 Best Practices

### 1. Start Simple

```bash
# Begin with basic configuration
python train_agent.py --algo ppo --config no_obstacle --timesteps 500000

# Then add complexity
python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000
```

### 2. Use Multiple Seeds

```bash
# Train with different seeds for robustness
for seed in 0 1 2 3 4; do
    python train_agent.py --algo ppo --config with_obstacle --seed $seed &
done
```

### 3. Proper Evaluation

```bash
# Always evaluate with more episodes
python evaluate_agent.py MODEL_PATH --algo ppo --n-episodes 50
```

### 4. Save Regularly

```bash
# Frequent checkpoints
python train_agent.py --algo ppo --save-freq 25000
```

### 5. Monitor Training

- Check TensorBoard regularly
- Watch for overfitting (train vs eval performance)
- Early stopping if no improvement

### 6. Hyperparameter Search

```bash
# Grid search example
for lr in 1e-4 3e-4 1e-3; do
    for bs in 64 128 256; do
        python train_agent.py --algo ppo --lr $lr --batch-size $bs
    done
done
```

## 📁 Directory Structure

After training, your workspace will look like:

```
project/
├── models/                          # Trained models
│   ├── PPO_with_obstacle_20251120/
│   │   ├── best_model/
│   │   │   └── best_model.zip      # Best model
│   │   ├── final_model.zip         # Final model
│   │   ├── vecnormalize.pkl        # Normalization stats
│   │   └── checkpoint_ppo_*.zip    # Checkpoints
│   └── SAC_with_obstacle_20251121/
│       └── ...
├── logs/                            # Training logs
│   └── PPO_with_obstacle_20251120/
│       ├── progress.csv
│       └── evaluations.npz
├── tensorboard/                     # TensorBoard logs
│   └── PPO_with_obstacle_20251120/
└── evaluation_results/              # Evaluation outputs
    ├── ppo_evaluation_results.json
    ├── ppo_evaluation_metrics.png
    └── ppo_sample_trajectories.png
```

## 🚨 Troubleshooting

### Issue: Training too slow

**Solutions:**
- Increase `n_envs` for parallel training
- Use GPU: `--device cuda`
- Reduce `n_steps` or `batch_size`
- Use faster algorithm (A2C, DDPG)

### Issue: Unstable training

**Solutions:**
- Reduce learning rate: `--lr 1e-4`
- Increase `n_envs` for more stable gradients
- Enable normalization (default)
- Try SAC (more stable than PPO)

### Issue: Poor performance

**Solutions:**
- Train longer: `--timesteps 2000000`
- Tune hyperparameters
- Check reward function
- Try different algorithm

### Issue: Out of memory

**Solutions:**
- Reduce `n_envs`
- Reduce `buffer_size` (for off-policy)
- Reduce `batch_size`
- Use CPU: `--device cpu`

## 📚 Additional Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [RL Algorithms Guide](https://spinningup.openai.com/)
- [Hyperparameter Tuning](https://github.com/DLR-RM/rl-baselines3-zoo)

## 🎓 Next Steps

1. **Train baseline model**:
   ```bash
   python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000
   ```

2. **Evaluate results**:
   ```bash
   python evaluate_agent.py models/.../best_model.zip --algo ppo --n-episodes 20
   ```

3. **Compare algorithms**:
   ```bash
   # Train multiple algorithms
   python train_agent.py --algo ppo --config with_obstacle &
   python train_agent.py --algo sac --config with_obstacle &
   python train_agent.py --algo td3 --config with_obstacle &
   ```

4. **Deploy best model** for real-world testing

---

**Happy Training! 🚀**

For questions or issues, please refer to the project documentation or create an issue.

