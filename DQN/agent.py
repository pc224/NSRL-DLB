import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models import DQN


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.LR)
        self.epsilon = config.EPSILON_START

    def select_action(self, state, mask):
        """
        使用 epsilon-greedy 策略选择动作，并严格应用 Mask 操作 (Section 4.2)
        """
        valid_actions = np.where(mask == 0)[0]
        if len(valid_actions) == 0:
            return 0

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(state_tensor).cpu().numpy()[0]

            q_values[mask == 1] = -1e9
            return np.argmax(q_values)

    def train_step(self, replay_buffer):
        if len(replay_buffer) < self.config.BATCH_SIZE:
            return 0.0

        state, action, reward, next_state, done, mask = replay_buffer.sample(self.config.BATCH_SIZE)

        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_tensor = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        done_tensor = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # 当前 Q 值
        q_values = self.online_net(state_tensor).gather(1, action_tensor)

        # 目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensor).max(1, keepdim=True)[0]
            target_q_values = reward_tensor + self.config.GAMMA * next_q_values * (1 - done_tensor)

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.config.EPSILON_MIN, self.epsilon * self.config.EPSILON_DECAY)