# 环境
from gym_env import GymEnv, BreakoutEnv, BreakoutStackEnv
env = BreakoutStackEnv()
state_dim = env.state_dim
action_dim = env.action_dim

# 算法
from dqn_agent import DQNAgent
conf = dict(
    action_dim = action_dim,
    eps_upper = 1,
    eps_lower = 0.02,
    eps_decay_freq = 1e6,
    gamma = 0.99,
    device = 'cuda',
    target_update_freq = 1000
)
agent = DQNAgent(conf)

# 模型
from q_network import QNetwork, CNN
model = CNN(action_dim, lr = 1e-4)
agent.set_model(model)


from sample import FrameNumpy, SampleBatchNumpy
from collections import deque
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

# 训练流程
buffer_size = 100000
batch_size = 32
episodes = 100000
train_returns = []
test_returns = []
replay_buffer = deque(maxlen = buffer_size) # 样本池
replay_initial = 1000

for episode in tqdm(range(episodes)):
    ret = 0
    obs = env.reset()
    agent_done = False
    while not agent_done:
        action = agent.predict(obs)
        next_obs, reward, real_done, agent_done, _ = env.step(action)
        ret += reward
        sample = FrameNumpy.from_dict({
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': agent_done
        })
        obs = next_obs
        # 每个step产生的样本加入样本池，并直接采样batch进行单次训练
        replay_buffer.append(sample)
        if len(replay_buffer) > replay_initial:
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            agent.learn(batch)
    train_returns.append((episode, ret))
    if episode % 100 == 0:
        # 每100局测试一局效果
        ret = 0
        obs = env.reset()
        real_done = False
        while not real_done:
            action = agent.exploit(obs) # 最优动作
            next_obs, reward, real_done, agent_done, _ = env.step(action)
            ret += reward
            obs = next_obs
        test_returns.append((episode, ret))

import pandas as pd
episodes = [ep for ep, _ret in train_returns]
train_ret = [ret for _ep, ret in train_returns]

df = pd.DataFrame({
    'episode': episodes,
    'train_return': train_ret
})

df.to_csv('train_returns.csv', index=False)

episodes = [ep for ep, _ret in test_returns]
test_ret = [ret for _ep, ret in test_returns]

df = pd.DataFrame({
    'episode': episodes,
    'train_return': test_ret
})

df.to_csv('test_returns.csv', index=False)


plt.plot([x[0] for x in train_returns], [x[1] for x in train_returns], label = 'train')
plt.plot([x[0] for x in test_returns], [x[1] for x in test_returns], label = 'test')
plt.legend()
plt.title("Breakout")
plt.show()
