import numpy as np
import sys
import math

# 定系为以岸边起点为原点的 东北地坐标系，
# 其中，x[0]>0向前（北），x[1]>0向右（东）,x[2]>0向下
# delta[0]垂直舵，控制航向——psi  delta[0]为正时，AUV受相对自身向左的力矩
# delta[2]水平舵，控制俯仰——theta  delta[2]为正时，AUV受相对自身下沉的力矩
# 动系为AUV以自身重心为原点的 前右下坐标系

class AuvModel:
    '''
    Define a AUV dynamic model class
    Input:
    x：State vector in fixed coordinate system，ndarray(6,1)
    u: Velocity vector in motion coordinate system，ndarray(6,1)
    delta: Rudder angle vector
    f: propeller
    '''
    p_water=1070#1000#density水密度
    L=2.38#length AUV长度 米

    m=167#mass AUV质量 kg
    g=9.81# gravity acceleration 重力加速度
    # position of the mass:重心坐标，一般定为运动坐标原点
    xg=0
    yg=0
    zg=0
    # position of the buoyant:浮心坐标，注意z方向与重心不重合
    xb=0
    yb=0
    zb=-0.06 # 浮心在重心之下-0.06

    #Inertia：刚体转动惯量项
    Ix=4.73 #  % 绕x轴的转动惯量
    Iy=70.27# % 绕x轴的转动惯量
    Iz=71.38# % 绕x轴的转动惯量

    # 以下是水动力系数，一大堆，看不懂，不过应该与具体型号的AUV有关，这些数值应该就是作者在实验中得来的
    x_u_=-4.8e-3
    y_r_=-9.67e-4
    y_v_=-3.66E-2
    y_r=-4.8E-3
    y_v=-0.0569
    y_delta_r=- 0.0215  #3.76e-4
    z_q_=-1.86e-4
    z_w_=-6.8e-2
    z_q=-0.0761
    z_w=-0.0830
    z_delta_s=-0.0330 # 5.76e-4
    k_vq=0
    k_wr=0
    m_q_=-4.1e-3
    m_w_=-1.82E-4
    m_q=-0.0207
    m_w=0.0372
    m_delta_s= 0.0213 # 3.72e-4 # 无因次流体动力阻尼系数
    n_r_=-2.4E-3
    n_v_=9.54E-4
    n_r=-9.67E-4
    n_v=-0.0186
    n_delta_r= 0.0099 # 无因次流体动力阻尼系数
    x_uu = -0.0093

    THRUST_MIN_AUV = 0  # 最小推力 
    THRUST_MAX_AUV = 500  # 最大推力
    RUDDER_MAX_AUV = 25*2*np.pi/360  # 舵角最大位置 (角度->弧度)


    def __init__(self,x,u,delta,f,t,random_factor_array, add_fuzzy_parameters):
        self.x = x
        self.u = u
        self.delta = delta
        self.f = f
        self.step = t*10

        self.phi=x[3].item()   #p
        self.theta=x[4].item() #q
        self.psi=x[5].item()  #r
        self.delta_r=delta[0].item()#产生Y，N delta_r--psi，航向，垂直舵
        self.delta_s=delta[2].item()#产生Z，M delta_s--theta，纵倾，水平舵
        

        self.u_=u[0].item() # 速度向量u的第1项，表示运动坐标系 航行（surge）线速度
        self.v=u[1].item()  # 速度向量u的第2项，表示运动坐标系 横向（sway）线速度
        self.w=u[2].item()  # 速度向量u的第3项，表示运动坐标系 生沉（heave）线速度
        self.p=u[3].item()  # 速度向量u的第4项，表示运动坐标系 横摇（roll）角速度
        self.q=u[4].item()  # 速度向量u的第5项，表示运动坐标系 俯仰（pitch）角速度
        self.r=u[5].item() # 速度向量u的第6项，表示运动坐标系 偏航（yaw）角速度
        self.tau = f

        self.add_fuzzy_parameters = add_fuzzy_parameters

        if self.add_fuzzy_parameters:
            self.random_factor_array = random_factor_array

            self.p_water = self.p_water * self.random_factor_array[0]#density水密度
            if round(self.step) % 20 == 0:
                self.m = self.m * np.random.uniform(0.7, 1.3)

            if round(self.step) % 100 == 0:
                exp = round(self.step) / 100
                self.p_water = self.p_water * (0.99**exp)

            # 以下是水动力系数，一大堆，看不懂，不过应该与具体型号的AUV有关，这些数值应该就是作者在实验中得来的
            self.x_u_ = self.x_u_ * self.random_factor_array[1]
            self.y_r_ = self.y_r_ * self.random_factor_array[2]
            self.y_v_ = self.y_v_ * self.random_factor_array[3]
            self.y_r = self.y_r * self.random_factor_array[4]
            self.y_v=self.y_v     * self.random_factor_array[5]
            self.y_delta_r=self.y_delta_r * self.random_factor_array[6]
            self.z_q_=self.z_q_ * self.random_factor_array[7]
            self.z_w_=self.z_w_ * self.random_factor_array[8]
            self.z_q=self.z_q   * self.random_factor_array[9]
            self.z_w=self.z_w   * self.random_factor_array[10]
            self.z_delta_s=self.z_delta_s * self.random_factor_array[11]
            self.k_vq=self.k_vq * self.random_factor_array[12]
            self.k_wr=self.k_wr * self.random_factor_array[13]
            self.m_q_=self.m_q_ * self.random_factor_array[14]
            self.m_w_=self.m_w_ * self.random_factor_array[15]
            self.m_q=self.m_q   * self.random_factor_array[16]
            self.m_w=self.m_w   * self.random_factor_array[17]
            self.m_delta_s=self.m_delta_s * self.random_factor_array[18]
            self.n_r_=self.n_r_ * self.random_factor_array[19]
            self.n_v_=self.n_v_ * self.random_factor_array[20]
            self.n_r=self.n_r   * self.random_factor_array[21]
            self.n_v=self.n_v   * self.random_factor_array[22]
            self.n_delta_r=self.n_delta_r * self.random_factor_array[23]
            self.x_uu =self.x_uu * self.random_factor_array[24]

    def M(self):
        M = np.array([
            [self.m-0.5*self.p_water*(self.L**3)*self.x_u_, 0, 0, 0, 0, 0],
            [0, self.m-0.5*self.p_water*(self.L**3)*self.y_v_, 0, 0, 0, -0.5*self.p_water*(self.L**4)*self.y_r_],
            [0, 0, self.m-0.5*self.p_water*(self.L**3)*self.z_w_, 0, -0.5*self.p_water*(self.L**4)*self.z_q_, 0],
            [0, 0, 0, self.Ix, 0, 0],
            [0, 0, -0.5*self.p_water*(self.L**4)*self.m_w_, 0, self.Iy-0.5*self.p_water*(self.L**5)*self.m_q_, 0],
            [0, -0.5*self.p_water*(self.L**4)*self.n_v_, 0, 0, 0, self.Iz-0.5*self.p_water*(self.L**5)*self.n_r_],
            ])
        
        # # 检查M矩阵中是否有异常值
        # if np.any(np.isnan(M)):
        #     print("异常检测: M矩阵包含NaN值")
        #     print("M矩阵:", M)
        #     print("相关参数: p_water={}, L={}, x_u_={}, y_v_={}, y_r_={}".format(
        #         self.p_water, self.L, self.x_u_, self.y_v_, self.y_r_))
        
        # if np.any(np.isinf(M)):
        #     print("异常检测: M矩阵包含Inf值")
        #     print("M矩阵:", M)
        
        # # 检查M矩阵是否接近奇异
        # try:
        #     cond_M = np.linalg.cond(M)
        #     if cond_M > 1e10:
        #         print("异常检测: M矩阵条件数过大: {}".format(cond_M))
        #         print("可能导致求逆不稳定")
        # except np.linalg.LinAlgError as e:
        #     print("异常检测: 无法计算M矩阵的条件数: {}".format(e))
        
        # # 检查M矩阵对角线元素是否接近零
        # for i in range(M.shape[0]):
        #     if np.abs(M[i, i]) < 1e-6:
        #         print("异常检测: M矩阵对角线元素[{},{}]接近零: {}".format(i, i, M[i, i]))
        
        return M

    def C(self):
        C = np.array([
            [0.5*self.p_water*(self.L**2)*self.x_uu*abs(self.u_), 0, 0, 0, self.m*self.w, -self.m*self.v],
            [0.5*self.p_water*(self.L**3)*self.y_r*self.r+0.5*self.p_water*(self.L**3)*self.y_v*self.v, 0, 0, -self.m*self.w, 0, self.m*self.u_],
            [0.5*self.p_water*(self.L**3)*self.z_q*self.q+0.5*self.p_water*(self.L**3)*self.z_w*self.w, 0, 0, self.m*self.v, -self.m*self.u_, 0],
            [0, 0.5*self.p_water*(self.L**4)*self.k_vq*self.q, 0.5*self.p_water*(self.L**4)*self.k_wr*self.r, 0, self.Iz*self.r, -self.Iy*self.q],
            [0.5*self.p_water*(self.L**4)*self.m_q*self.q+0.5*self.p_water*(self.L**3)*self.m_w*self.w, 0, 0, -self.Iz*self.r, 0, self.Ix*self.p],
            [0.5*self.p_water*(self.L**4)*self.n_r*self.r+0.5*self.p_water*(self.L**4)*self.n_v*self.v, 0, 0, self.Iy*self.q, -self.Ix*self.p, 0],
            ])
        # 科氏向心力矩阵
        return C

    def G(self):
        G = np.array([
            [0],
            [0],
            [0],
            [-self.m*self.g*(self.zg-self.zb)*np.cos(self.theta)*np.sin(self.phi)],
            [-self.m*self.g*(self.zg-self.zb)*np.sin(self.theta)],
            [0]
            ])
        return G
        # gravity重力与浮力的力与力矩矩阵。此仿真中假设了重力与浮力抵消，
        # 重心坐标[xg,yg,zg]、浮心坐标[xb,yb,zb]在水平面重合，在垂直方向不重合，本仿真中浮心在重心下0.06m     
        #tau=f#thruster推进器
    #以下的矩阵不是很明确，虽然rudder的翻译是舵，但看起来像是流体阻尼矩阵，也估计是一个意思，流体阻尼在舵面的作用下为AUV提供了操纵
    def Rudder(self):
        Rudder = np.array([
            [0], 
            [0.5*self.p_water*(self.L**2)*self.y_delta_r*(self.u_**2)*self.delta_r],
            [0.5*self.p_water*(self.L**2)*self.z_delta_s*(self.u_**2)*self.delta_s],
            [0],
            [0.5*self.p_water*(self.L**3)*self.m_delta_s*(self.u_**2)*self.delta_s],
            [0.5*self.p_water*(self.L**3)*self.n_delta_r*(self.u_**2)*self.delta_r]
            ])
        return Rudder

    def get_du(self, x, u, delta, f):
        self.x = x
        self.u = u
        self.delta = delta
        self.f = f

        # # 首先检查输入中是否有异常值
        # if np.any(np.isnan(x)):
        #     print("异常检测: 状态x包含NaN值: {}".format(x))
        # if np.any(np.isinf(x)):
        #     print("异常检测: 状态x包含Inf值: {}".format(x))
        # if np.any(np.isnan(u)):
        #     print("异常检测: 速度u包含NaN值: {}".format(u))
        # if np.any(np.isinf(u)):
        #     print("异常检测: 速度u包含Inf值: {}".format(u))
        
        # 提取状态值
        self.phi = x[3].item()
        self.theta = x[4].item()
        self.psi = x[5].item()
        self.delta_r = delta[0].item()
        self.delta_s = delta[2].item()
        
        self.u_ = u[0].item()
        self.v = u[1].item()
        self.w = u[2].item()
        self.p = u[3].item()
        self.q = u[4].item()
        self.r = u[5].item()
        
        # 构建各矩阵
        M = self.M()
        C = self.C()
        Rudder = self.Rudder()
        G = self.G()
        
        # # 检查C矩阵
        # if np.any(np.isnan(C)) or np.any(np.isinf(C)):
        #     print("异常检测: C矩阵包含异常值:")
        #     print("C =", C)
        #     print("状态值: u_={}, v={}, w={}, p={}, q={}, r={}".format(
        #         self.u_, self.v, self.w, self.p, self.q, self.r))
        
        # # 检查舵面力矩阵
        # if np.any(np.isnan(Rudder)) or np.any(np.isinf(Rudder)):
        #     print("异常检测: Rudder矩阵包含异常值:")
        #     print("Rudder =", Rudder)
        #     print("舵角: delta_r={}, delta_s={}".format(self.delta_r, self.delta_s))
        
        # # 检查重力矩阵
        # if np.any(np.isnan(G)) or np.any(np.isinf(G)):
        #     print("异常检测: G矩阵包含异常值:")
        #     print("G =", G)
        #     print("姿态角: phi={}, theta={}, psi={}".format(self.phi, self.theta, self.psi))
        
        # 计算右侧项并检查
        C_u = np.matmul(C, u)
        # if np.any(np.isnan(C_u)) or np.any(np.isinf(C_u)):
        #     print("异常检测: C*u计算结果包含异常值:")
        #     print("C*u =", C_u)
        
        right_term = C_u + Rudder + f + G
        # if np.any(np.isnan(right_term)) or np.any(np.isinf(right_term)):
        #     print("异常检测: 加速度计算右侧项包含异常值:")
        #     print("右侧项 =", right_term)
        #     print("C*u =", C_u)
        #     print("Rudder =", Rudder)
        #     print("f =", f)
        #     print("G =", G)
        
        # 尝试计算M的逆矩阵
        M_inv = np.linalg.inv(M)
        # try:
        #     M_inv = np.linalg.inv(M)
            
        #     if np.any(np.isnan(M_inv)) or np.any(np.isinf(M_inv)):
        #         print("异常检测: M逆矩阵包含异常值:")
        #         print("M_inv =", M_inv)
        #         print("M =", M)
        # except np.linalg.LinAlgError as e:
        #     print("异常检测: M矩阵求逆失败: {}".format(e))
        #     print("M =", M)
        #     # 如果无法求逆，可以尝试使用伪逆进行诊断
        #     try:
        #         M_pinv = np.linalg.pinv(M)
        #         print("使用伪逆的结果:")
        #         print("M_pinv =", M_pinv)
        #     except Exception as e2:
        #         print("伪逆计算也失败: {}".format(e2))
        #     # 返回一个零加速度向量，或原始向量以继续执行
        #     return np.zeros_like(u)
        
        # 计算最终加速度
        du = np.matmul(M_inv, right_term)
        
        # # 检查加速度结果
        # if np.any(np.isnan(du)):
        #     print("异常检测: 计算的加速度包含NaN值:")
        #     print("du =", du)
        # if np.any(np.isinf(du)):
        #     print("异常检测: 计算的加速度包含Inf值:")
        #     print("du =", du)
        
        # 检查加速度是否过大
        # max_expected_accel = 1e3  # 设置一个大的但合理的阈值
        # if np.any(np.abs(du) > max_expected_accel):
        #     print("异常检测: 加速度值异常大:")
        #     print("du =", du)
        #     print("超过阈值的索引:", np.where(np.abs(du) > max_expected_accel))
            
        #     # 进一步诊断原因
        #     print("诊断加速度异常原因:")
        #     print("M_inv 范数:", np.linalg.norm(M_inv))
        #     print("right_term 范数:", np.linalg.norm(right_term))
        #     print("M 行列式:", np.linalg.det(M))
        
        return du
  
    def radlimit(self,rad): # this function is used to limit a input rad in [-pi,pi]
        count = 0
        un_converge = False
        while True:         
            if rad > np.pi:                
                rad = rad - 2*np.pi
                count += 1 
                if count >= 1e3:                 
                    #print("The rad doesn't converge")                 
                    #sys.exit()   
                    un_converge = True  
                    break                              
            elif rad < -np.pi:
                rad = rad +2*np.pi
                count += 1 
                if count >= 1e3:                                      
                    #print("The rad doesn't converge")                                      
                    #sys.exit()
                    un_converge = True 
                    break
            else:
                break
        return rad, un_converge
        

class CoordinateTrans:
    def __init__(self,x,u):
        self.x = x
        self.u = u
        self.phi=x[3].item()   #p
        self.theta=x[4].item() #q
        self.psi=x[5].item()  #r

    def mov_to_fix(self,x,u): #   

        self.x = x
        self.u = u
        self.phi = self.x[3].item()   
        self.theta = self.x[4].item() 
        self.psi = self.x[5].item()  

        inv_T = np.array([             
            [np.cos(self.psi)*np.cos(self.theta),np.cos(self.psi)*np.sin(self.theta)*np.sin(self.phi)-np.sin(self.psi)*np.cos(self.phi),np.cos(self.psi)*np.sin(self.theta)*np.cos(self.phi)+np.sin(self.psi)*np.sin(self.phi),0,0,0],             
            [np.sin(self.psi)*np.cos(self.theta),np.sin(self.psi)*np.sin(self.theta)*np.sin(self.phi)+np.cos(self.psi)*np.cos(self.phi),np.sin(self.psi)*np.sin(self.theta)*np.cos(self.phi)-np.cos(self.psi)*np.sin(self.phi),0,0,0], 
            [-np.sin(self.theta),np.cos(self.theta)*np.sin(self.phi),np.cos(self.phi)/np.cos(self.theta),0,0,0],             
            [0,0,0,1,np.sin(self.phi)*np.tan(self.theta),np.cos(self.phi)*np.tan(self.theta)],             
            [0,0,0,0,np.cos(self.phi),-np.sin(self.phi)],             
            [0,0,0,0,np.sin(self.phi)/np.cos(self.theta),np.cos(self.phi)/np.cos(self.theta)] # np无sec()方法，所以*sec()全部被替换为/cos()
            ])         
        return np.matmul(inv_T,self.u)     
