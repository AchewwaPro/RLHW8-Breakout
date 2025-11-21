# 环境
from gym_env import GymEnv, BreakoutEnv, BreakoutStackEnv
env = BreakoutStackEnv()
state_dim = env.state_dim
action_dim = env.action_dim

import gym
try:
    action_meanings = env.env.unwrapped.get_action_meanings()
    FIRE_ACTION = next((i for i, m in enumerate(action_meanings) if 'FIRE' in m), None)
    print('action_meanings:', action_meanings, 'FIRE_ACTION =', FIRE_ACTION)
except Exception as e:
    print('Warning: cannot get action meanings, FIRE will be disabled:', e)
    FIRE_ACTION = None

# 算法
from dqn_agent import DQNAgent
conf = dict(
    action_dim = action_dim,
    eps_upper = 0.9,
    eps_lower = 0.1,
    eps_decay_freq = 5e5,
    random_step = 0,
    gamma = 0.99,
    device = 'cuda',
    target_update_freq = 3000
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
buffer_size = 500000
batch_size = 64
episodes = 10000
train_returns = []
test_returns = []
replay_buffer = deque(maxlen = buffer_size) # 样本池
replay_initial = 10000
update_freq = 4

import time

for episode in tqdm(range(episodes)):
    ret = 0
    obs = env.reset()
    agent_done = False
    real_done = False

    # 开局自动发球
    if FIRE_ACTION is not None:
        obs, _, real_done, agent_done, info = env.step(FIRE_ACTION)
    # print(FIRE_ACTION, _, real_done, agent_done)
    # time.sleep(1)
    while not real_done:
        action = agent.predict(obs)
        next_obs, reward, real_done, agent_done, _ = env.step(action)
        # print(action, reward, real_done, agent_done)
        # time.sleep(1)
        ret += reward
        sample = FrameNumpy.from_dict({
            'obs': obs,
            'next_obs': next_obs,
            'action': action,
            'reward': reward,
            'done': real_done
        })
        obs = next_obs
        agent.global_step += 1
        # 每个step产生的样本加入样本池，并直接采样batch进行单次训练
        replay_buffer.append(sample)
        if len(replay_buffer) > replay_initial and agent.global_step % update_freq == 0:
            batch = random.sample(replay_buffer, batch_size)
            batch = SampleBatchNumpy.stack(batch)
            agent.sample_process(batch)
            agent.learn(batch)

        # 死亡一次但没结束，自动发球
        if FIRE_ACTION is not None and not real_done and agent_done:
            obs, _, real_done, agent_done, info = env.step(FIRE_ACTION)

    train_returns.append((episode, ret))
    if episode % 50 == 0:
        print(f"eps={agent.eps}")
        print(f"learn_step={agent.learn_step}")
        ret = 0
        obs = env.reset()
        real_done = False
        if FIRE_ACTION is not None:
            obs, _, real_done, agent_done, info = env.step(FIRE_ACTION)
        while not real_done:
            action = agent.exploit(obs) # 最优动作
            next_obs, reward, real_done, agent_done, _ = env.step(action)
            ret += reward
            obs = next_obs
            if FIRE_ACTION is not None and not real_done and agent_done:
                obs, _, real_done, agent_done, info = env.step(FIRE_ACTION)
        test_returns.append((episode, ret))
        print(f"reward={ret}")

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
