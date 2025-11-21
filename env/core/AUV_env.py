import math
import gym 
from gym import spaces 
import numpy as np
import models.AUV_model_fuzzy as model
from env.utils.RandomTraj3D import RandomTraj3D

"""
modified env: 
1. 初始艏向角 pi/2， 初始舵角 0， 初始推力 0
2. quxiao限制 x,y 线速度和 艏向 角速度；z轴线速度、横滚角速度、俯仰角速度恒为0
3. x、y坐标 艏向角无限制， z坐标恒为20， 横滚角、俯仰角恒为0， 艏向角范围保持在(-pi,pi)
4. step()中实现 “动作饱和设置”
5. 去掉前进奖励
6. 删去了一些没必要的代码
此版本的任务目标为使AUV跟踪一个正弦线路径y = (50/math.pi)*(np.sin((math.pi/50)*x))+50
"""
class AUVEnv(gym.Env):
   
    def __init__(self):
        # 动作饱和限制
        self.min_f = model.AuvModel.THRUST_MIN_AUV
        self.max_f = model.AuvModel.THRUST_MAX_AUV
        self.max_delta = model.AuvModel.RUDDER_MAX_AUV

        # 奖励相关
        self.reward = 0  
        self.step_counter = 0
        self.total_steps = 0
        # 距离
        # self.d_terminal = None
        self.d_path = None 

        self.un_converge = False # 动力学模型是否发散（False表示收敛）

        self.x = np.zeros((6,1))
        self.u = np.zeros((6,1))
        self.begin_random_train = True
        self.start = self.x[:3]
        self.delta = np.zeros((3,1))  # 初始舵角 （为角度值）
        self.f = np.zeros((6,1))  # 初始化推进器推力
        self.t = 0  # 仿真时间归零
        self.dt = 0.1  # 仿真时间步长

        self.x_desire = 0
        self.y_desire = 0
        self.z_desire = 0
        self.desired_yaw = 0
        self.desired_pitch = 0

        self.add_fuzzy_parameters = False  # 是否使用模糊化水动力参数

        self.min_action = np.array([self.min_f, -self.max_delta, -self.max_delta])
        self.max_action = np.array([self.max_f, self.max_delta, self.max_delta])
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(3,), dtype=np.float32)
        
        self.reset(episodes=0)

    def reset(self, episodes):
        '''重置一次，并返回观察'''
        self.env_episodes = episodes
        # print('env-episodes:',self.env_episodes)
        self.t = 0  # 仿真时间归零
        self.dt = 0.1  # 仿真时间步长

        # self.d_terminal = None
        self.d_path = None 

        self.reward = 0  
        self.step_counter = 0 # 一轮仿真的步数计数归零

        RJ = RandomTraj3D()
        self.x_traj, self.y_traj, self.z_traj = RJ.rand_traj()
        self.ref_traj = np.array([self.x_traj, self.y_traj, self.z_traj])

        self.x_desire = 0
        self.y_desire = 0
        self.z_desire = 0
        self.desired_yaw = 0
        self.desired_pitch = 0

        self.u = np.zeros((6,1))
        self.x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(6,1) # 每一轮初始位置坐标及航向角
        self.delta = np.zeros((3,1))  # 初始舵角
        self.f = np.zeros((6,1))  # 初始化推进器推力

        if self.begin_random_train:  # 以下为随机初始化起点
            temp = np.random.randint(0, 2)
            if temp == 0: self.x[0] += np.random.rand(1)
            else: self.x[0] -= np.random.rand(1)

            temp = np.random.randint(0, 2)
            if temp == 0: self.x[1] += np.random.rand(1)
            else: self.x[1] -= np.random.rand(1)

            temp = np.random.randint(0, 2)
            if temp == 0: self.x[-1] += (45/180*np.pi) * np.random.rand(1)
            else: self.x[-1] -= (45/180*np.pi) * np.random.rand(1)  # 及随机的艏向角psi

            # random start velocity
            temp = np.random.randint(0, 2)
            if temp == 0: self.u[0] = 1 + np.random.rand(1)*0.5
            else: self.u[0] = 1 - np.random.rand(1)*0.5

            temp = np.random.randint(0, 2)
            if temp == 0: self.u[1] += np.random.rand(1)*0.1
            else: self.u[1] -= np.random.rand(1)*0.1

            temp = np.random.randint(0, 2)
            if temp == 0: self.u[-1] += np.random.rand(1)*0.1
            else: self.u[-1] -= np.random.rand(1)*0.1

            # random f delta
            self.f[0] = np.random.rand(1)*500

            temp = np.random.randint(0, 2)
            if temp == 0: self.delta[0] += np.random.rand(1) * (25*np.pi/180)
            else: self.delta[0] -= np.random.rand(1) * (25*np.pi/180)

        self.start = self.x[:3]
        self.un_converge = False # 动力学模型是否发散（False表示收敛）

        # 使用模糊化参数生成AUV动力学模型
        if self.add_fuzzy_parameters == True: # 2000~2999
            self.random_factor_array = np.random.uniform(0.8, 1.2, 25) # 每个水动力系数都各自 乘 0.9~1.1的随机系数，以此达到模糊化的目的
            print("fuzzy training ...")
        else:
            self.random_factor_array = np.ones(25)

        return self.get_observe().squeeze()

    def step(self, action):  # 动作空间的大小，没归一化
        action_temp = action
        self.f[0] = action_temp[0]
        self.delta[0] = action_temp[1]
        self.delta[2] = action_temp[2]
        self.x, self.u = self.runge_kutta() # 时刻t 更新AUV状态向量、速度向量 增加了一个动力学模型收敛性判断
        step_reward, d_path = self.step_reward() # 根据上一时刻的期望位置、角度和当前时刻的位置、角度得到奖励

        self.t += self.dt
        self.step_counter += 1  # 每轮的步数计数

        done = False
        if self.step_counter > 999:
            done = True
            # self.env_episodes +=1

        obs = self.get_observe() # 获得新的观察 时刻t+1
    
        if self.un_converge:
            print("Model nonconvergence!")    
        # done = self.un_converge or done
        if done:
            print('Ending point:', self.x[0],self.x[1],self.x[2])
        return  np.hstack(obs), step_reward, done, d_path, self.ref_traj

    def runge_kutta(self): # 四阶龙格库塔法得到 新的速度向量u、状态向量x,并对角度做出限制
        # Auv = model.AuvModel(self.x,self.u,self.delta,self.f)
        self.Auv = model.AuvModel(self.x,self.u,self.delta,self.f, self.t, self.random_factor_array, self.add_fuzzy_parameters)
        k1 = self.Auv.get_du(self.x, self.u, self.delta, self.f)
        k2 = self.Auv.get_du(self.x, self.u + 0.5*self.dt*k1, self.delta,self.f)
        k3 = self.Auv.get_du(self.x, self.u + 0.5*self.dt*k2, self.delta,self.f)
        k4 = self.Auv.get_du(self.x, self.u + self.dt*k3, self.delta,self.f)
        # 更新速度向量u
        self.u = self.u + (self.dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        self.u[3] = 0
        
        # 动系下的速度向量坐标变化为定系下的速度向量
        Trans = model.CoordinateTrans(self.x,self.u) # 实例化一个坐标变换矩阵  
        dx = Trans.mov_to_fix(self.x, self.u) # 得到定系下的速度向量
        x = dx * self.dt + self.x # 得到新的状态向量

        x[3] = 0
        x[4] = radlimit(x[4])
        x[5] = radlimit(x[5])

        self.x = x  # 经过AUV六自由度动力学模型计算得到一个时间步后新的状态向量x   

        return self.x, self.u#, un_converge 

    def get_observe(self):
        obs = np.zeros((17,1)) # ***改变obs须调节此处***

        self.x_desire = self.x_traj[self.step_counter]
        self.y_desire = self.y_traj[self.step_counter]
        self.z_desire = self.z_traj[self.step_counter]

        delta_x = self.x[0] - self.x_desire  # x当前与期望位置差
        delta_y = self.x[1] - self.y_desire  # y当前与期望位置差
        delta_z = self.x[2] - self.z_desire
        self.d_path = (delta_x ** 2 + delta_y ** 2 + delta_z ** 2) ** 0.5

        x_d = self.x_traj[self.step_counter+1] - self.x_traj[self.step_counter]
        y_d = self.y_traj[self.step_counter+1] - self.y_traj[self.step_counter]
        z_d = self.z_traj[self.step_counter+1] - self.z_traj[self.step_counter]

        # 水平面
        alpha_k = np.arctan2(y_d, x_d)  # 路径切角
        # 垂直面(x, z)和(y, z) vertical
        beta_k = np.arctan(-z_d/(np.sqrt(x_d**2 + y_d**2)))

        R1 = np.array([
            [np.cos(alpha_k), -np.sin(alpha_k), 0],
            [np.sin(alpha_k), np.cos(alpha_k), 0],
            [0, 0, 1],
        ])
        R2 = np.array([
            [np.cos(beta_k), 0, np.sin(beta_k)],
            [0, 1, 0],
            [-np.sin(beta_k), 0, np.cos(beta_k)],
        ])
        R = np.dot(R1, R2).T  # 转换矩阵
        P = np.array([delta_x, delta_y, delta_z])  # 路径坐标系下的误差（当前的期望路径点为原点）
        Pe = np.dot(R, P)  # 求出定系下的误差
        ye = Pe[1].item()
        ze = Pe[2].item()

        delta1 = 2.38 * 2  # AUV长度 look-ahead distance
        delta2 = (ye ** 2 + delta1 ** 2) ** 0.5

        xp = alpha_k
        vp = beta_k
        xr = np.arctan((-ye) / delta1)
        vr = np.arctan((ze) / delta2)
        self.desired_yaw = xp + xr
        self.desired_pitch = vp + vr

        obs[0] = self.x[0] - self.x_desire
        obs[1] = self.x[1] - self.y_desire
        obs[2] = self.x[2] - self.z_desire

        obs[3] = self.d_path

        obs[4] = self.u[0]
        obs[5] = self.u[1]
        obs[6] = self.u[2]
        obs[7] = self.u[4]
        obs[8] = self.u[5]

        obs[9] = np.cos(self.desired_pitch)
        obs[10] = np.sin(self.desired_pitch)
        obs[11] = np.cos(self.desired_pitch - self.x[-2])
        obs[12] = np.sin(self.desired_pitch - self.x[-2])
        obs[13] = np.cos(self.desired_yaw)
        obs[14] = np.sin(self.desired_yaw)
        obs[15] = np.cos(self.desired_yaw - self.x[-1])
        obs[16] = np.sin(self.desired_yaw - self.x[-1])

        return obs.squeeze()
        
    def step_reward(self): # 定义单步奖励
        done = False
        # self.total_steps += 1

        step_reward = 0 # 重置单步奖励

        # 靠近奖励
        d_path = (
            (self.x_desire - self.x[0])**2 +
            (self.y_desire - self.x[1])**2 +
            (self.z_desire - self.x[2])**2
            ) ** 0.5
        step_reward -= np.clip(np.log2(d_path + 1e-20), -1, 1)

        # 俯仰角奖励
        d_pitch = radlimit(self.desired_pitch - self.x[4])
        step_reward += np.cos(d_pitch)

        # 偏航角奖励
        d_yaw = radlimit(self.desired_yaw - self.x[5])
        step_reward += np.cos(d_yaw)

        return step_reward, d_path

def radlimit(rad):  # this function is used to limit a input rad in [-pi,pi]
    count = 0
    while True:
        if rad > np.pi:
            rad = rad - 2 * np.pi
            count += 1
            if count >= 1e3:
                break
        elif rad < -np.pi:
            rad = rad + 2 * np.pi
            count += 1
            if count >= 1e3:
                break
        else:
            break
    return rad