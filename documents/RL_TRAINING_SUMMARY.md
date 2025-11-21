# 🎓 RL Training System Summary

完整的多算法强化学习训练系统，支持 AUV 障碍物避障任务。

## 📦 已创建的文件

### 训练脚本
1. **`train_agent.py`** - 通用训练脚本
   - 支持 5 种算法：PPO, SAC, TD3, DDPG, A2C
   - 完整的回调系统（评估、检查点、TensorBoard）
   - 自动超参数配置
   - 观察归一化
   - 学习率调度

2. **`train_ppo.py`** - PPO 专用训练脚本（原始版本）
   - 保留用于向后兼容

### 评估脚本
3. **`evaluate_agent.py`** - 通用评估脚本
   - 支持所有训练的算法
   - 生成详细的评估报告
   - 自动生成可视化图表
   - 保存轨迹数据

4. **`evaluate_model.py`** - 模型评估脚本（简化版）

### 文档
5. **`documents/TRAINING_GUIDE.md`** - 完整训练指南
   - 所有算法的详细说明
   - 超参数调优指南
   - 最佳实践
   - 故障排除

6. **`README_TRAINING.md`** - 快速入门指南
   - 简洁的使用说明
   - 常见命令
   - 快速示例

### 配置文件
7. **`requirements_rl.txt`** - RL 依赖包
   - Stable-Baselines3
   - TensorBoard
   - 其他必需库

### 辅助脚本
8. **`scripts/quick_train.sh`** - Linux/Mac 快速训练脚本
9. **`scripts/quick_train.bat`** - Windows 快速训练脚本

## 🤖 支持的算法

### 1. PPO (Proximal Policy Optimization)
**类型**: On-Policy  
**特点**:
- ✅ 最稳定可靠
- ✅ 适合初学者
- ✅ 样本效率适中
- ✅ 易于调参

**默认超参数**:
```python
learning_rate: 3e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95
clip_range: 0.2
```

**使用场景**: 首选算法，适合大多数任务

### 2. SAC (Soft Actor-Critic)
**类型**: Off-Policy  
**特点**:
- ✅ 非常稳定
- ✅ 样本效率高
- ✅ 自动熵调节
- ⚠️ 训练速度较慢

**默认超参数**:
```python
learning_rate: 3e-4
buffer_size: 1_000_000
batch_size: 256
tau: 0.005
gamma: 0.99
ent_coef: 'auto'
```

**使用场景**: 需要高样本效率的任务

### 3. TD3 (Twin Delayed DDPG)
**类型**: Off-Policy  
**特点**:
- ✅ 高效稳定
- ✅ 适合连续控制
- ✅ 训练速度快
- ⚠️ 需要调参

**默认超参数**:
```python
learning_rate: 3e-4
buffer_size: 1_000_000
batch_size: 256
policy_delay: 2
target_policy_noise: 0.2
```

**使用场景**: 需要快速训练的任务

### 4. DDPG (Deep Deterministic Policy Gradient)
**类型**: Off-Policy  
**特点**:
- ✅ 训练快速
- ✅ 实现简单
- ⚠️ 稳定性较差
- ⚠️ 对超参数敏感

**使用场景**: 快速原型开发

### 5. A2C (Advantage Actor-Critic)
**类型**: On-Policy  
**特点**:
- ✅ 训练最快
- ✅ 内存占用小
- ⚠️ 样本效率低
- ⚠️ 性能一般

**使用场景**: 快速测试和验证

## 📊 算法对比表

| 特性 | PPO | SAC | TD3 | DDPG | A2C |
|------|-----|-----|-----|------|-----|
| **稳定性** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **样本效率** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **训练速度** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## 🎯 使用示例

### 基础训练

```bash
# PPO (推荐)
python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000

# SAC (高样本效率)
python train_agent.py --algo sac --config with_obstacle --timesteps 1000000

# TD3 (快速稳定)
python train_agent.py --algo td3 --config with_obstacle --timesteps 1000000

# DDPG (快速原型)
python train_agent.py --algo ddpg --config with_obstacle --timesteps 500000

# A2C (快速测试)
python train_agent.py --algo a2c --config with_obstacle --timesteps 500000
```

### 高级配置

```bash
# 自定义学习率和批量大小
python train_agent.py --algo ppo --lr 1e-4 --batch-size 128

# 使用 GPU
python train_agent.py --algo sac --device cuda

# 多个并行环境
python train_agent.py --algo ppo --n-envs 8

# 禁用观察归一化
python train_agent.py --algo td3 --no-normalize

# 禁用学习率调度
python train_agent.py --algo ppo --no-linear-schedule
```

### 评估

```bash
# 基础评估
python evaluate_agent.py models/PPO_*/best_model/best_model.zip --algo ppo

# 带可视化
python evaluate_agent.py models/SAC_*/best_model/best_model.zip --algo sac --render

# 更多 episode
python evaluate_agent.py models/TD3_*/best_model/best_model.zip --algo td3 --n-episodes 50
```

## 📁 输出结构

训练后的文件组织：

```
project/
├── models/                                    # 训练模型
│   ├── PPO_with_obstacle_20251120_120000/
│   │   ├── best_model/
│   │   │   └── best_model.zip               # ⭐ 最佳模型
│   │   ├── final_model.zip                  # 最终模型
│   │   ├── vecnormalize.pkl                 # 归一化统计
│   │   └── checkpoint_ppo_*.zip             # 检查点
│   ├── SAC_with_obstacle_20251120_130000/
│   │   ├── best_model/
│   │   ├── final_model.zip
│   │   ├── vecnormalize.pkl
│   │   └── replay_buffer.pkl                # 重放缓冲
│   └── ...
├── logs/                                      # 训练日志
│   └── PPO_with_obstacle_20251120_120000/
│       ├── progress.csv                     # 训练进度
│       └── evaluations.npz                  # 评估结果
├── tensorboard/                               # TensorBoard 日志
│   └── PPO_with_obstacle_20251120_120000/
│       └── PPO_1/
│           └── events.out.tfevents.*
└── evaluation_results/                        # 评估输出
    ├── ppo_evaluation_results.json          # 数值结果
    ├── ppo_evaluation_metrics.png           # 指标图表
    └── ppo_sample_trajectories.png          # 轨迹图
```

## 🔧 关键特性

### 1. 统一接口
所有算法使用相同的命令行接口：
```bash
python train_agent.py --algo [ALGORITHM] --config [CONFIG] [OPTIONS]
```

### 2. 自动超参数
每个算法都有优化的默认超参数，可以直接使用或覆盖。

### 3. 完整回调系统
- **EvalCallback**: 定期评估并保存最佳模型
- **CheckpointCallback**: 定期保存检查点
- **TensorboardCallback**: 记录自定义指标
- **ProgressBarCallback**: 显示训练进度

### 4. 观察归一化
自动使用 `VecNormalize` 进行观察和奖励归一化。

### 5. 学习率调度
支持线性学习率衰减。

### 6. 并行训练
支持多个并行环境加速训练（On-Policy 算法）。

### 7. GPU 支持
自动检测 CUDA 或手动指定设备。

### 8. 详细日志
- CSV 日志
- TensorBoard 日志
- 控制台输出

## 📈 训练监控

### TensorBoard 指标

**通用指标**:
- `rollout/ep_rew_mean`: 平均 episode 奖励
- `rollout/ep_len_mean`: 平均 episode 长度
- `time/fps`: 训练速度（FPS）

**自定义指标**:
- `rollout/ep_final_error`: 最终路径误差
- `rollout/ep_collision`: 碰撞率
- `rollout/ep_path_error`: 路径跟踪误差
- `rollout/ep_obstacle_distance`: 最小障碍物距离
- `rollout/ep_path_reward`: 路径奖励
- `rollout/ep_obstacle_reward`: 避障奖励

**算法特定指标**:
- PPO: `train/entropy_loss`, `train/policy_gradient_loss`
- SAC: `train/ent_coef`, `train/actor_loss`
- TD3: `train/actor_loss`, `train/critic_loss`

### 查看训练进度

```bash
# 启动 TensorBoard
tensorboard --logdir tensorboard/

# 在浏览器打开
http://localhost:6006
```

## 🎓 最佳实践

### 1. 算法选择

**初学者**: 使用 PPO
```bash
python train_agent.py --algo ppo --config with_obstacle
```

**需要高效率**: 使用 SAC
```bash
python train_agent.py --algo sac --config with_obstacle
```

**需要快速训练**: 使用 TD3 或 A2C
```bash
python train_agent.py --algo td3 --config with_obstacle
```

### 2. 训练策略

**阶段 1 - 基础训练** (500K-1M steps):
```bash
python train_agent.py --algo ppo --config no_obstacle --timesteps 500000
```

**阶段 2 - 添加障碍物** (1M-2M steps):
```bash
python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000
```

**阶段 3 - 完整挑战** (2M-3M steps):
```bash
python train_agent.py --algo ppo --config hard --timesteps 2000000
```

### 3. 超参数调优

**学习率调整**:
- 过快收敛 → 降低学习率
- 学习太慢 → 增加学习率
- 推荐范围: 1e-5 到 1e-3

**批量大小调整**:
- 内存不足 → 减小批量
- 训练不稳定 → 增大批量
- PPO 推荐: 64-256
- SAC 推荐: 256-512

### 4. 评估策略

```bash
# 定期评估
python evaluate_agent.py MODEL_PATH --algo ALGO --n-episodes 20

# 最终评估
python evaluate_agent.py MODEL_PATH --algo ALGO --n-episodes 50
```

### 5. 多种子训练

```bash
# 训练多个种子以验证稳定性
for seed in 0 1 2 3 4; do
    python train_agent.py --algo ppo --seed $seed &
done
```

## 🚨 常见问题

### Q: 哪个算法最好？
**A**: PPO 是最稳定可靠的选择。SAC 样本效率更高但训练较慢。

### Q: 需要训练多久？
**A**: 
- 基础性能: 500K-1M steps
- 良好性能: 1M-2M steps
- 最佳性能: 2M-3M steps

### Q: GPU 必须吗？
**A**: 不必须，但强烈推荐。GPU 可以加速 3-5 倍。

### Q: 如何判断训练是否成功？
**A**: 查看评估奖励是否稳定提升，碰撞率是否降低。

### Q: 内存不足怎么办？
**A**: 
```bash
# 减少并行环境
python train_agent.py --algo ppo --n-envs 2

# 减小批量大小
python train_agent.py --algo ppo --batch-size 32

# 使用 CPU
python train_agent.py --algo ppo --device cpu
```

## 📚 相关文档

- 📖 [完整训练指南](TRAINING_GUIDE.md)
- 🏗️ [环境使用指南](obstacle_avoidance_env_guide.md)
- 🔧 [集成总结](INTEGRATION_SUMMARY.md)
- 🐛 [Bug 修复说明](../BUGFIX_NOTES.md)

## ✅ 下一步

1. **安装依赖**:
   ```bash
   pip install -r requirements_rl.txt
   ```

2. **快速训练**:
   ```bash
   python train_agent.py --algo ppo --config with_obstacle --timesteps 1000000
   ```

3. **监控训练**:
   ```bash
   tensorboard --logdir tensorboard/
   ```

4. **评估模型**:
   ```bash
   python evaluate_agent.py models/.../best_model.zip --algo ppo --n-episodes 20
   ```

5. **对比算法**:
   ```bash
   python train_agent.py --algo ppo --config with_obstacle &
   python train_agent.py --algo sac --config with_obstacle &
   python train_agent.py --algo td3 --config with_obstacle &
   ```

---

**系统状态**: ✅ 完整且经过测试  
**最后更新**: 2025-11-20  
**版本**: v1.0.0

