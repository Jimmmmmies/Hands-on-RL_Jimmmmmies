import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from configs import moving_average

class PolicyNet(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = F.softmax(self.fc2(x), dim=1) # The output is a probability distribution over actions
        return x
    
class REINFORCE:
    
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.pnet = PolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.pnet.parameters(), lr=self.lr)
        
    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.pnet(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        state_list = transition_dict['state']
        action_list = transition_dict['action']
        reward_list = transition_dict['reward']
        
        G = 0
        self.optimizer.zero_grad(set_to_none=True)
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            G = self.gamma * G + reward
            prob = torch.log(self.pnet(state).gather(1, action))
            loss = -prob * G
            loss.backward()
        self.optimizer.step()
        
lr = 2e-3
n_episode = 2000
hidden_dim = 128
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
env = gym.make('CartPole-v1')
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = REINFORCE(state_dim, action_dim, hidden_dim, lr, gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(n_episode / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(n_episode / 10)):
            episode_return = 0
            transition_dict = {
                'state': [], 
                'action': [],
                'next_state': [],
                'reward': [],
                'done': []
                }
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transition_dict['state'].append(state)
                transition_dict['action'].append(action)
                transition_dict['next_state'].append(next_state)
                transition_dict['reward'].append(reward)
                transition_dict['done'].append(done)
                state = next_state
                episode_return += reward
            agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (n_episode / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('REINFORCE on CartPole-v1')
plt.show()

mv_return = moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('REINFORCE on CartPole-v1')
plt.show()