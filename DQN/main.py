import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from config import Config
from data_generator import generate_dataset, get_fixed_test_case
from environment import DisassemblyEnv
from replay_buffer import ReplayBuffer
from agent import DQNAgent


def train_agent():
    cfg = Config()

    print(f"正在生成训练数据集 (固定任务数 N={cfg.N_TASKS}, 对应论文 Section 4.7)...")
    dataset = generate_dataset(cfg.DATASET_SIZE, cfg.N_TASKS, cfg.CYCLE_TIME)

    TP, times = dataset[0]
    env = DisassemblyEnv(len(times), cfg.CYCLE_TIME, TP, times)
    agent = DQNAgent(env.state_dim, env.action_dim, cfg)
    replay_buffer = ReplayBuffer(cfg.MEMORY_SIZE)

    rewards_history = []
    losses_history = []
    total_steps = 0

    print(f"开始训练 DQN (State Dim: {env.state_dim}, Action Dim: {env.action_dim})...")
    for episode in range(cfg.EPISODES):
        TP, times = dataset[np.random.randint(0, len(dataset))]
        env = DisassemblyEnv(len(times), cfg.CYCLE_TIME, TP, times)

        state, mask = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, mask)
            next_state, reward, done, info, next_mask = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done, next_mask)
            state = next_state
            mask = next_mask

            loss = agent.train_step(replay_buffer)
            if loss > 0:
                losses_history.append(loss)

            total_steps += 1
            episode_reward += reward

            if total_steps % cfg.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode: {episode + 1}/{cfg.EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_history, losses_history


def test_and_evaluate(agent, cfg):

    print(f"\n--- 开始测试固定案例 (N={cfg.N_TASKS}) ---")
    TP, times = get_fixed_test_case(cfg.N_TASKS, cfg.CYCLE_TIME, cfg.TEST_CASE_SEED)
    env = DisassemblyEnv(cfg.N_TASKS, cfg.CYCLE_TIME, TP, times)

    state, mask = env.reset()
    done = False
    sequence = []
    ws_assignments = {}

    start_time = time.time()
    while not done:
        agent.epsilon = 0.0
        action = agent.select_action(state, mask)
        sequence.append(action + 1)

        next_state, reward, done, info, next_mask = env.step(action)

        # 根据环境返回的实际工作站索引进行记录，彻底杜绝空工作站和任务错位
        actual_ws = info['ws_index']
        if actual_ws not in ws_assignments:
            ws_assignments[actual_ws] = []
        ws_assignments[actual_ws].append(action + 1)

        state = next_state
        mask = next_mask

    end_time = time.time()
    running_time = end_time - start_time

    # 计算目标函数值 (Eq. 1)
    obj_value = 0
    for ws, tasks in ws_assignments.items():
        ws_time = sum(times[t - 1] for t in tasks)
        obj_value += (cfg.CYCLE_TIME - ws_time) ** 2

    print(f"运行时间 (Running time): {running_time:.4f} s")
    print(f"目标函数值 (Idle time balancing index): {obj_value}")
    print(f"激活工作站数量 (Activated workstations): {len(ws_assignments)}")
    print(f"拆卸序列 (Disassembly sequence): {sequence}")
    print("工作站分配方案 (Workstation assignments):")
    for ws, tasks in ws_assignments.items():
        print(f"  W{ws}: {tasks}")


def plot_results(rewards, losses):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Episode Reward', alpha=0.6)
    window = min(50, len(rewards))
    if window > 0:
        plt.plot(np.convolve(rewards, np.ones(window) / window, mode='valid'), label=f'Moving Avg ({window})',
                 color='red')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', alpha=0.6)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')

    plt.tight_layout()
    plt.savefig('dqn_training_results.png')
    print("\n训练曲线已保存至 'dqn_training_results.png'")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    agent, rewards, losses = train_agent()
    test_and_evaluate(agent, Config())
    plot_results(rewards, losses)