from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F
import copy
import math

from sample import Frame, SampleBatchNumpy

# DQN算法
class DQNAgent(Agent):

    def __init__(self, conf):
        # ε-贪心算法所需的参数
        self.action_dim = conf['action_dim']
        self.eps_lower = conf.get('eps_lower', 0.1)
        self.eps_upper = conf.get('eps_upper', 1)
        self.eps_decay_freq = conf.get('eps_decay_freq', 200)
        self.epsilon = self.eps_upper
        # 累计回报的衰减系数
        self.gamma = conf.get('gamma', 0.95)
        # 设备信息，如'cpu', 'cuda:0'等
        self.device = conf.get('device', 'cpu')
        self.target_update_freq = conf.get('target_update_freq', 100)
        self.learn_step = 0

    def epsilon_decay(self, total_step):
        self.eps = self.eps_lower + (self.eps_upper - self.eps_lower) * math.exp(-total_step / self.eps_decay_freq)
    
    # 设置推理和训练使用的模型
    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.target_model = copy.deepcopy(model)
        self.target_model.to(self.device)

    # 输入单帧状态，采样探索性动作
    def predict(self, obs: 'Frame | np.ndarray | dict'):
        obs = Frame.convert(obs)
        if np.random.random() < self.epsilon:
            # 以 ε 的概率随机选择动作
            action = np.random.randint(0, self.action_dim)
        else:
            # 以 1-ε 的概率选择Q值最大的动作
            obs = obs.to_torch(device = self.device)
            q_value = self.model.inference(obs)
            action = int(q_value.argmax())
        return action
    
    # 输入单帧状态，计算最优动作
    def exploit(self, obs: 'Frame | np.ndarray | dict'):
        # 直接选择Q值最大的动作
        obs = Frame.convert(obs)
        obs = obs.to_torch(device = self.device)
        q_value = self.model.inference(obs)
        action = int(q_value.argmax())
        return action
    
    # 输入样本Batch，训练模型
    def learn(self, samples: SampleBatchNumpy):
        '''
        samples需要包含字段: obs, next_obs, action, reward, done
        '''
        samples = samples.to_torch(device = self.device)
        # 计算Q(s, a)
        q_values = self.model.inference(samples.obs)
        # print(q_values.shape, samples.action.shape)
        q_value = q_values.gather(1, samples.action)

        with torch.no_grad():
            next_q_values = self.model.inference(samples.next_obs)
            next_action = next_q_values.max(dim=1, keepdim=True)[1]
            next_q_values_target = self.target_model.inference(samples.next_obs)
            next_q_value_target = next_q_values_target.gather(1, next_action)
            # Q函数更新目标
            q_target = samples.reward + self.gamma * (1 - samples.done) * next_q_value_target
        
        loss = F.smooth_l1_loss(q_value, q_target)
        # 模型更新
        self.model.train(loss)

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.epsilon_decay(self.learn_step)

    
    def sample_process(self, samples: SampleBatchNumpy):
        '''
        samples需要包含字段: obs, next_obs, action, reward, done
        '''
        samples.action = samples.action.reshape((-1, 1)).astype(np.int64)
        samples.reward = samples.reward.reshape((-1, 1)).astype(np.float32)
        samples.done = samples.done.reshape((-1, 1)).astype(np.float32)
        