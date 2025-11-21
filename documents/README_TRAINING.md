
# 🤖 AUV Obstacle Avoidance - RL Training

Train reinforcement learning agents to control AUV trajectory tracking with obstacle avoidance.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_rl.txt
```

### 2. Train Your First Agent

```bash
# Quick training with PPO (recommended for beginners)
python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000

# Or use the quick training script
bash scripts/quick_train.sh ppo with_obstacle 1000000 4 auto
```

### 3. Monitor Training

```bash
# Start TensorBoard
tensorboard --logdir tensorboard/

# Open browser to: http://localhost:6006
```

### 4. Evaluate Trained Model

```bash
# Find your model in models/ directory
python evaluate_agent.py models/PPO_with_obstacle_*/best_model/best_model.zip \
    --algo ppo --config with_obstacle --n-episodes 20 --render
```

## 🎯 Supported Algorithms

| Algorithm | Command | Best For | Speed | Stability |
|-----------|---------|----------|-------|-----------|
| **PPO** | `--algo ppo` | General use | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SAC** | `--algo sac` | Sample efficiency | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **TD3** | `--algo td3` | High-dimensional | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **DDPG** | `--algo ddpg` | Fast prototyping | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **A2C** | `--algo a2c` | Quick testing | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## 📝 Training Examples

### Example 1: PPO (Recommended)

```bash
python train_agent.py \
    --algo ppo \
    --config with_obstacle \
    --timesteps 1000000 \
    --n-envs 4 \
    --device auto
```

### Example 2: SAC (More Sample Efficient)

```bash
python train_agent.py \
    --algo sac \
    --config with_obstacle \
    --timesteps 1000000 \
    --n-envs 1 \
    --device cuda
```

### Example 3: TD3 (Fast and Stable)

```bash
python train_agent.py \
    --algo td3 \
    --config training \
    --timesteps 2000000 \
    --n-envs 1 \
    --device cuda
```

### Example 4: Hard Mode Challenge

```bash
python train_agent.py \
    --algo sac \
    --config hard \
    --timesteps 3000000 \
    --n-envs 1 \
    --device cuda
```

## 🎮 Environment Configurations

| Config | Obstacle | Ocean Current | Difficulty |
|--------|----------|---------------|------------|
| `no_obstacle` | ❌ | ❌ | ⭐ Easy |
| `with_obstacle` | ✅ | ❌ | ⭐⭐⭐ Medium |
| `with_current` | ✅ | ✅ | ⭐⭐⭐⭐ Hard |
| `training` | ✅ | ❌ + Fuzzy | ⭐⭐⭐⭐ Hard |
| `hard` | ✅ | ✅ + Fuzzy | ⭐⭐⭐⭐⭐ Very Hard |

## 🔧 Common Commands

### Training

```bash
# Basic training
python train_agent.py --algo ppo --config with_obstacle

# With custom hyperparameters
python train_agent.py --algo ppo --lr 1e-4 --batch-size 128

# Multiple parallel environments
python train_agent.py --algo ppo --n-envs 8

# Long training
python train_agent.py --algo sac --timesteps 5000000
```

### Evaluation

```bash
# Basic evaluation
python evaluate_agent.py MODEL_PATH --algo ppo --config with_obstacle

# With rendering
python evaluate_agent.py MODEL_PATH --algo ppo --render

# More episodes for better statistics
python evaluate_agent.py MODEL_PATH --algo ppo --n-episodes 50
```

### Monitoring

```bash
# TensorBoard
tensorboard --logdir tensorboard/

# Specific run
tensorboard --logdir tensorboard/PPO_with_obstacle_20251120/
```

## 📊 Expected Results

### PPO after 1M steps:
- Average Reward: ~500-700
- Success Rate: ~80-90%
- Final Path Error: <1.0m

### SAC after 1M steps:
- Average Reward: ~600-800
- Success Rate: ~85-95%
- Final Path Error: <0.8m

### TD3 after 1M steps:
- Average Reward: ~550-750
- Success Rate: ~82-92%
- Final Path Error: <0.9m

## 🐛 Troubleshooting

### Issue: Training is slow
**Solution:**
```bash
# Use more parallel environments
python train_agent.py --algo ppo --n-envs 8

# Use GPU
python train_agent.py --algo ppo --device cuda
```

### Issue: Out of memory
**Solution:**
```bash
# Reduce parallel environments
python train_agent.py --algo ppo --n-envs 2

# Use CPU
python train_agent.py --algo ppo --device cpu
```

### Issue: Unstable training
**Solution:**
```bash
# Use SAC (more stable)
python train_agent.py --algo sac --config with_obstacle

# Reduce learning rate
python train_agent.py --algo ppo --lr 1e-4
```

## 📁 Output Structure

```
project/
├── models/                          # Trained models
│   └── PPO_with_obstacle_20251120/
│       ├── best_model/             # Best model (use this!)
│       ├── final_model.zip
│       └── vecnormalize.pkl
├── logs/                            # Training logs
├── tensorboard/                     # TensorBoard logs
└── evaluation_results/              # Evaluation outputs
```

## 📚 Documentation

- 📖 [Complete Training Guide](documents/TRAINING_GUIDE.md)
- 🏗️ [Environment Guide](documents/obstacle_avoidance_env_guide.md)
- 🔧 [Integration Summary](documents/INTEGRATION_SUMMARY.md)

## 🎓 Learning Resources

- [Stable-Baselines3 Tutorial](https://stable-baselines3.readthedocs.io/)
- [RL Zoo Examples](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)

## 💡 Tips

1. **Start with PPO** - Most reliable for beginners
2. **Use TensorBoard** - Monitor training progress
3. **Train multiple seeds** - Check consistency
4. **Evaluate thoroughly** - Use 20+ episodes
5. **Save frequently** - Don't lose progress

## 🏆 Advanced Usage

### Hyperparameter Tuning

```bash
# Grid search
for lr in 1e-4 3e-4 1e-3; do
    python train_agent.py --algo ppo --lr $lr &
done
```

### Multi-Seed Training

```bash
# Train with 5 different seeds
for seed in 0 1 2 3 4; do
    python train_agent.py --algo ppo --seed $seed &
done
```

### Resume Training

```bash
# Load and continue training
# (Coming soon - model checkpoints can be loaded)
```

## 🤝 Contributing

Found a bug or have a suggestion? Please create an issue!

## 📧 Contact

For questions or support, please refer to the project documentation.

---

**Happy Training! 🚀**

Start with: `python train_agent.py --algo ppo --config with_obstacle`

