# ⚡ Quick Start - RL Training

## 🎯 One-Liner Training

```bash
python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000
```

## 📦 Setup (First Time Only)

```bash
pip install -r requirements_rl.txt
```

## 🤖 Algorithm Cheat Sheet

| Want | Use | Command |
|------|-----|---------|
| 🏆 **Best Overall** | PPO | `--algo ppo` |
| ⚡ **Most Efficient** | SAC | `--algo sac` |
| 🚀 **Fastest** | TD3/A2C | `--algo td3` |
| 🎯 **Most Stable** | SAC | `--algo sac` |

## 🎮 Quick Commands

### Train
```bash
# PPO (Recommended)
python train_agent.py --algo ppo --config with_obstacle

# SAC (High Efficiency)
python train_agent.py --algo sac --config with_obstacle

# TD3 (Fast & Stable)
python train_agent.py --algo td3 --config with_obstacle
```

### Evaluate
```bash
python evaluate_agent.py models/PPO_*/best_model/best_model.zip --algo ppo --n-episodes 20
```

### Monitor
```bash
tensorboard --logdir tensorboard/
```

## 📊 Configs

| Config | Difficulty | Command |
|--------|-----------|---------|
| No Obstacle | ⭐ | `--config no_obstacle` |
| With Obstacle | ⭐⭐⭐ | `--config with_obstacle` |
| + Ocean Current | ⭐⭐⭐⭐ | `--config with_current` |
| Hard Mode | ⭐⭐⭐⭐⭐ | `--config hard` |

## 🔧 Common Options

```bash
--algo ppo              # Algorithm (ppo/sac/td3/ddpg/a2c)
--config with_obstacle  # Environment
--timesteps 1000000     # Training steps
--n-envs 4             # Parallel envs
--device cuda          # Use GPU
--lr 3e-4              # Learning rate
--seed 0               # Random seed
```

## 📁 Find Your Model

```
models/PPO_with_obstacle_YYYYMMDD_HHMMSS/
└── best_model/
    └── best_model.zip  ← Use this!
```

## 🎓 Full Documentation

- 📖 [Training Guide](documents/TRAINING_GUIDE.md)
- 🤖 [System Summary](documents/RL_TRAINING_SUMMARY.md)
- 📘 [README Training](README_TRAINING.md)

## 💡 Pro Tips

1. Start with PPO: `python train_agent.py --algo ppo`
2. Use TensorBoard: `tensorboard --logdir tensorboard/`
3. Evaluate often: `--eval-freq 10000`
4. Save frequently: `--save-freq 50000`
5. Use GPU: `--device cuda`

---

**Need help?** Check [TRAINING_GUIDE.md](documents/TRAINING_GUIDE.md)

