@echo off
REM Quick training script for Windows

setlocal enabledelayedexpansion

echo ========================================
echo AUV Obstacle Avoidance - Quick Training
echo ========================================

REM Parse arguments with defaults
set ALGO=%1
if "%ALGO%"=="" set ALGO=ppo

set CONFIG=%2
if "%CONFIG%"=="" set CONFIG=with_obstacle

set TIMESTEPS=%3
if "%TIMESTEPS%"=="" set TIMESTEPS=1000000

set N_ENVS=%4
if "%N_ENVS%"=="" set N_ENVS=4

set DEVICE=%5
if "%DEVICE%"=="" set DEVICE=auto

echo.
echo Configuration:
echo   Algorithm: %ALGO%
echo   Environment: %CONFIG%
echo   Timesteps: %TIMESTEPS%
echo   Parallel Envs: %N_ENVS%
echo   Device: %DEVICE%
echo ========================================
echo.

REM Run training
python train_agent.py ^
    --algo %ALGO% ^
    --config %CONFIG% ^
    --timesteps %TIMESTEPS% ^
    --n-envs %N_ENVS% ^
    --device %DEVICE% ^
    --eval-freq 10000 ^
    --n-eval-episodes 5 ^
    --save-freq 50000 ^
    --verbose 1

echo.
echo ========================================
echo Training completed!
echo Check models\ directory for saved models
echo View training progress: tensorboard --logdir tensorboard\
echo ========================================

endlocal

