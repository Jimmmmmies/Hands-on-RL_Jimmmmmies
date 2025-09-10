import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
from tqdm import tqdm
from configs import moving_average

class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done
    
    def return_size(self):
        return len(self.buffer)
    
class QNetwork(nn.Module):
    
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class DQN:
    
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.device = device
        self.qnet = QNetwork(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_qnet = QNetwork(state_dim, hidden_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.qnet.parameters(), lr=lr) # Only update qnet
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.qnet(state).argmax().item()
        return action
    
    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # Double DQN, deal with overestimation
        # max_action = self.qnet(next_state).max(1)[1].view(-1, 1)
        # max_q = self.target_qnet(next_state).gather(1, max_action)
        q = self.qnet(state).gather(1, action)
        max_q = self.target_qnet(next_state).max(1)[0].view(-1, 1)
        target_q = reward + self.gamma * max_q * (1 - done)
        dqn_loss = torch.mean(F.mse_loss(q, target_q))
        self.optimizer.zero_grad(set_to_none=True)
        dqn_loss.backward()
        self.optimizer.step()
        
        if self.count % self.target_update == 0: # Update target network by copying weights from qnet
            self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.count += 1
        
env = gym.make('CartPole-v1')
random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
n_episodes = 1000
gamma = 0.98
epsilon = 0.01
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
hidden_dim = 128
lr = 4e-3
buffer_size = 10000
batch_size = 64
minimal_size = 500
target_update = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
agent = DQN(state_dim, action_dim, hidden_dim, lr, gamma, epsilon, target_update, device)
replay_buffer = ReplayBuffer(buffer_size)

return_list = []
for i in range(10):
    with tqdm(total=int(n_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(n_episodes / 10)):
            episode_return = 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                episode_return += reward
                state = next_state
                if replay_buffer.return_size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'state' : b_s,
                        'action' : b_a,
                        'reward' : b_r,
                        'next_state' : b_ns,
                        'done' : b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (n_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
            
episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('DQN on CartPole-v1')
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episode_list, mv_return)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('DQN on CartPole-v1')
plt.show()