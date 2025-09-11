import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from configs import rl_utils

class PolicyNet(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x
    
class ValueNet(nn.Module):
    
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
class ActorCritic:
    
    def __init__(self, state_dim, action_dim, hidden_dim, policy_lr, action_lr, gamma, device):
        self.gamma = gamma
        self.device = device
        self.policy_lr = policy_lr
        self.action_lr = action_lr
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device) # Sarsa with value function approximation
        self.policy_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.policy_lr)
        self.value_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.action_lr)
        
    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        prob = self.actor(state)
        action_dist = torch.distributions.Categorical(prob)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)
        
        # Update Critic
        target_q = reward + self.gamma * self.critic(next_state) * (1 - done)
        delta = target_q - self.critic(state)
        # Update Actor
        prob = self.actor(state).gather(1, action)
        log_prob = torch.log(prob)
        
        actor_loss = torch.mean(-log_prob * delta.detach()) # Use mean because batch update
        critic_loss = torch.mean(F.mse_loss(self.critic(state), target_q.detach()))
        self.policy_optimizer.zero_grad(set_to_none=True)
        self.value_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        critic_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
actor_lr = 1e-3
critic_lr = 1e-2 # Usually the critic has a larger learning rate
n_episode = 2000
hidden_dim = 128
gamma = 0.98
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = gym.make('CartPole-v1')
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, device)
return_list = rl_utils.train_on_policy_agent(env, agent, n_episode)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Actor-Critic on CartPole-v1')
plt.show()

mv_return_list = rl_utils.moving_average(return_list, 9)
plt.plot(episode_list, mv_return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Actor-Critic on CartPole-v1')
plt.show()