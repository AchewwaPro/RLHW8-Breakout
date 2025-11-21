from env import Env
import gym

class GymEnv(Env):

    def __init__(self, name, render_mode=None):
        if render_mode:
            self.env = gym.make(name, render_mode="human")
        else:
            self.env = gym.make(name)
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

    def reset(self, conf = {}):
        obs = self.env.reset()
        # 旧版gym接口返回单个obs，新版接口返回(obs, info)
        if type(obs) == tuple: obs = obs[0]
        return obs
    
    def step(self, action):
        # 旧版gym接口step返回(obs, reward, done, info)，新版接口返回(obs, reward, done, truncated, info)
        obs, reward, *t = self.env.step(action)
        if len(t) == 2:
            return obs, reward, t[0], t[1]
        else:
            done, truncated, info = t
            return obs, reward, done or truncated, info

import numpy as np
class BreakoutEnv(Env):

    def __init__(self):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.state_dim = (80, 80)
        self.action_dim = 4
    
    def preprocess_obs(self, obs):
        # 为了训练效率，对状态进行简化
        obs = np.mean(obs, axis = 2) # 灰度图
        obs = obs[35:195] # 裁剪中间区域
        obs = obs[::2, ::2] # 下采样
        obs = obs.astype(np.float32) / 255
        obs = obs[np.newaxis, :, :]
        return obs
    
    def reset(self, conf = {}):
        obs = self.env.reset()
        # 旧版gym接口返回单个obs，新版接口返回(obs, info)
        if type(obs) == tuple: obs = obs[0]
        return self.preprocess_obs(obs)
    
    def step(self, action):
        obs, reward, *t = self.env.step(action)
        obs = self.preprocess_obs(obs)
        if len(t) == 2:
            return obs, reward, t[0], t[1]
        else:
            done, truncated, info = t
            return obs, reward, done or truncated, info
        
from collections import deque
        
class BreakoutStackEnv(Env):

    def __init__(self, stack_size=4):
        self.env = gym.make("BreakoutDeterministic-v4")
        self.state_dim = (stack_size, 80, 80)
        self.action_dim = 4
        self.stack = deque(maxlen=stack_size)
        self.stack_size = stack_size
        self.lives = 0
    
    def preprocess_obs(self, obs):
        # 为了训练效率，对状态进行简化
        obs = np.mean(obs, axis = 2) # 灰度图
        obs = obs[35:195] # 裁剪中间区域
        obs = obs[::2, ::2] # 下采样
        obs = obs.astype(np.float32) / 256
        # obs = obs[np.newaxis, :, :]
        return obs

    def reset(self, conf = {}):
        obs = self.env.reset()
        # 旧版gym接口返回单个obs，新版接口返回(obs, info)
        if type(obs) == tuple: obs = obs[0]
        obs = self.preprocess_obs(obs)
        self.stack = deque([obs] * self.stack_size, maxlen=self.stack_size)
        
        if hasattr(self.env.unwrapped, 'ale'):
            self.lives = self.env.unwrapped.ale.lives()

        return np.stack(self.stack, axis=0)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.preprocess_obs(obs)

        real_done = done or truncated
        if hasattr(self.env.unwrapped, 'ale'):
            lives = self.env.unwrapped.ale.lives()
        else:
            lives = info.get('lives', 0)
        
        life_lost = (lives < self.lives) and (lives > 0)
        # print(self.lives, lives)
        self.lives = lives

        self.stack.append(obs)
        agent_done = real_done or life_lost

        if real_done:
            pass
        elif life_lost:
            # self.stack = deque([obs] * self.stack_size, maxlen=self.stack_size)
            pass
        # reward = np.sign(reward).astype(np.float32)
        return np.stack(self.stack, axis=0), reward, real_done, agent_done, info
    


        