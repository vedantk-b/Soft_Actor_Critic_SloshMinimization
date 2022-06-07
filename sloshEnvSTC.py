from argparse import Action
# from cv2 import FlannBasedMatcher
import numpy as np
from stable_baselines3 import PPO
import gym 
from gym import spaces
from numpy import sin, cos, power
import matplotlib.pyplot as plt
import random

from scipy.integrate import odeint

sign = lambda z: abs(z)/z if z!=0 else 0  # One line equivalent of the signum function
	
class SloshEnv  (gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, k1= 1.776,k2= 4.084):
        super(SloshEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=10, shape=(3,))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low = -100, high = 100, shape=(4,), dtype=np.float64)

        self.k1 = k1
        self.k2 = k2
        self.ms = 1.32 # in kg
        self.M = 10.82 # M=Ml+Mc+Mb+m=10.82
        self.g = 9.81
        self.c1 = 1
        self.c2 = 11
        self.cd = 0.00030490 # damping coefficient given in equation (1b)
        self.l = 0.052126    #in meter
        self.d = 0        # d is external disturbance in input channel equation 1(a)

    def surface(self, zeta1, zeta2):
        return self.c1 * zeta2 + self.c2 *zeta1

    def dynamics(self, state, u1):
        # equation 4 decomposition of states

        zeta1 = state[0]
        sigman = state[1]
        zeta3 = state[2]
        zeta4 = state[3]
        z = state[4]

        # Using self.[variables] to facilitate future dynamical parameter updates
        # Decompose back to normal variable names for ease of writing code
        k1 = self.k1
        k2 = self.k2
        ms = self.ms
        M = self.M
        g = self.g
        c1 = self.c1
        c2 = self.c2
        cd = self.cd
        l = self.l
        d = self.d
        D = M - ms * power(cos(zeta3),2)


        # __________ EQUATION 5 ____________

        f1_term1 = ms * g * sin(zeta3) * cos(zeta3)
        f1_term2 = (cd/l) * zeta4 * cos(zeta3)
        f1_term3 = ms * l * power(zeta4,2) * sin(zeta3)

    
        f1 = (f1_term1 + f1_term2 + f1_term3)/D
        b1 = 1/D



        b1Nominal = 1/(M-(ms/2))
        deltaB1 = b1 - b1Nominal

        wl_bar = (c2*(sigman-c2*zeta1))/(c1) 
        

        vl = -k1 * power(abs(sigman),1/2) * sign(sigman) + z


        u = (vl - wl_bar)/(c1 * b1Nominal)
        u1.append(u)

        de = c1 * f1 + c1 * b1 * d + c1 * deltaB1 * u


        
        zeta1Dot = (sigman - (c2*zeta1))/(c1)
        sigmanDot = -k1 * power(abs(sigman),1/2) * sign(sigman) + z + de
        # print(sigmanDot)
        zeta3Dot = zeta4 

        bigStuff = vl + de - (((c2)/(c1)) * (sigman - (c2*zeta1)))
        term1 = -(cos(zeta3)/(l*c1)) * bigStuff
        term2 =  - (g/l) * sin(zeta3) - (cd/(ms * power(l,2))) * zeta4

        zeta4Dot = term1 + term2
        zDot = -k2 * sign(sigman) 

#        print([zeta1Dot,sigmanDot,zeta3Dot,zeta4Dot,zDot])
        return [zeta1Dot,sigmanDot,zeta3Dot,zeta4Dot,zDot]

    def step(self, action):
        T = 0.1
        # print(f"initial states:\n x = {self.x}\n xdot = {self.xDot}\n phi = {self.phi}\n phiDot = {self.phiDot}\n")
        self.c2 = action[0]
        self.k1 = action[1]
        self.k2 = action[2]
        xDes = 0.175
        xDotDes = 0
        phiDes = 0
        phiDotDes = 0

        ex = self.x - xDes
        exDot = self.xDot - xDotDes
        ephi = self.phi - phiDes
        ephiDot = self.phiDot - phiDotDes

        initCon = [ex,self.surface(ex,exDot),ephi,ephiDot,0]

        dT = 0.0001          
        x = np.array(initCon)
        sol = []
        u1=[]   # control input
        # cost = []
        for t in np.arange(0,T,dT):
            sol.append(x)
            xDot = self.dynamics(x,u1)
            xDot = np.array(xDot)
#            print(xDot)
            x = x + xDot*dT
            cost = -(4*abs(x[0])+0.3*abs(x[2])+0.3*abs(x[3]))
            self.total_reward += cost
            
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
        sol = np.array(sol)

        return_state = [0,0,0,0]

        return_state[0] = sol[-1][0]+xDes
        return_state[1] = xDot[0]
        return_state[2] = sol[-1][2] 
        return_state[3] = sol[-1][3] 

        if(abs(return_state[0] - 0.175) < 0.0001):
            self.done = True
    
        # if(plot):
        #     self.plot(t,sol,u1)
        
        info = {"sol":sol}

        self.x = return_state[0]
        self.xDot = return_state[1]
        self.phi = return_state[2]
        self.phiDot = return_state[3]
            
        observation = np.array(return_state)
        
        return observation, self.reward, self.done, info

        
    def reset(self):
        self.total_reward = 0
        self.prev_reward  = 0
        self.done = False
        self.x = 0.0
        self.xDot = 0.0
        self.phi = 0.0
        self.phiDot = 0.0
        observation = [0.0, 0.0, 0.0, 0.0]
        observation = np.array(observation)
        return observation
