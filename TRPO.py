import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import copy
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
    
class TRPO:
    
    def __init__(self, state_dim, action_dim, hidden_dim, kl_constraint,
                 value_lr, gamma, lmbda, alpha, device):
        self.device = device
        self.actor = PolicyNet(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=value_lr)
        self.gamma = gamma
        self.lmbda = lmbda # GAE parameter
        self.kl_constraint = kl_constraint
        self.alpha = alpha # Step size of Line Search
        
    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        prob = self.actor(state)
        action_dist = torch.distributions.Categorical(prob)
        action = action_dist.sample()
        return action.item()
    
    def hessian_matrix_vector_product(self, state, old_action_dist, vector):
        # Calculate the product of the Hessian matrix of KL divergence and a vector
        new_action_dist = torch.distributions.Categorical(self.actor(state))
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))
        # Calculate the first order gradient g of KL divergence
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # Calculate the second order gradient of g^T * v, which is Hv
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector
        
    def conjugate_gradient(self, grad, state, old_action_dist, max_iter=10):
        # Solve Hx = g using Conjugate Gradient where H is the Hessian matrix of KL divergence
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(max_iter):
            Hp = self.hessian_matrix_vector_product(state, old_action_dist, p)
            alpha = rdotr / (torch.dot(p, Hp))
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def compute_surrogate_obj(self, state, action, advantage, old_log_prob, actor):
        log_prob = torch.log(actor(state).gather(1, action))
        ratio = torch.exp(log_prob - old_log_prob)
        return torch.mean(ratio * advantage)
    
    def line_search(self, state, action, advantage, old_log_prob, old_action_dist, max_vec, max_iter=15):
        old_parameters = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(state, action, advantage, old_log_prob, self.actor)
        
        for i in range(max_iter):
            coefficient = self.alpha ** i
            new_parameters = old_parameters + coefficient * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_parameters, new_actor.parameters())
            new_action_dist = torch.distributions.Categorical(new_actor(state))
            kl_divergence = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))
            new_obj = self.compute_surrogate_obj(state, action, advantage, old_log_prob, new_actor)
            # Update condition: improve objective and satisfy KL constraint
            if new_obj > old_obj and kl_divergence < self.kl_constraint:
                return new_parameters
        return old_parameters
    
    def policy_learn(self, state, action, old_action_dist, old_log_prob, advantage):
        # Update policy network
        surrogate_obj = self.compute_surrogate_obj(state, action, advantage, old_log_prob, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # Solving Hx = g using Conjugate Gradient
        descent_direction = self.conjugate_gradient(obj_grad, state, old_action_dist)
        Hx = self.hessian_matrix_vector_product(state, old_action_dist, descent_direction)
        max_step = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hx) + 1e-8))
        new_parameters = self.line_search(state, action, advantage, old_log_prob, old_action_dist,
                                          max_step * descent_direction)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_parameters, self.actor.parameters())
        
    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)
        
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_error = td_target - self.critic(state)
        # Using GAE to compute advantage estimates
        # .cpu() is to convert tensor from pytorch to numpy for compute_advantage function
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)
        old_log_prob = torch.log(self.actor(state).gather(1, action)).detach()
        # The probability of the old strategy used in old_action_dist should not be used to 
        # calculate the gradient with the current parameters. Thus, we detach it.
        old_action_dist = torch.distributions.Categorical(self.actor(state).detach())
        # Update value network
        critic_loss = torch.mean(F.mse_loss(self.critic(state), td_target.detach()))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update policy network
        self.policy_learn(state, action, old_action_dist, old_log_prob, advantage)
        
class PolicyNetContinuous(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * F.tanh(self.fc_mu(x)) # Use tanh to satisfy the action range of the environment
        std = F.softplus(self.fc_std(x)) # Use softplus to ensure std is positive
        return mu, std
    
class TRPOContinuous:
    
    def __init__(self, state_dim, action_dim, hidden_dim, kl_constraint,
                 value_lr, gamma, lmbda, alpha, device):
        self.device = device
        self.actor = PolicyNetContinuous(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=value_lr)
        self.gamma = gamma
        self.lmbda = lmbda # GAE parameter
        self.kl_constraint = kl_constraint
        self.alpha = alpha # Step size of Line Search
        
    def choose_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return action.cpu().numpy().flatten()
    
    def hessian_matrix_vector_product(self, state, old_action_dist, vector, damping=0.1):
        # Calculate the product of the Hessian matrix of KL divergence and a vector
        mu, std = self.actor(state)
        new_action_dist = torch.distributions.Normal(mu, std)
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))
        # Calculate the first order gradient g of KL divergence
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # Calculate the second order gradient of g^T * v, which is Hv
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        # .contiguous() is to ensure the memory space is continuous for view operation
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        # Use damping to make sure the Hessian matrix is positive definite
        return grad2_vector + damping * vector
        
    def conjugate_gradient(self, grad, state, old_action_dist, max_iter=10):
        # Solve Hx = g using Conjugate Gradient where H is the Hessian matrix of KL divergence
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        
        for i in range(max_iter):
            Hp = self.hessian_matrix_vector_product(state, old_action_dist, p)
            alpha = rdotr / (torch.dot(p, Hp))
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x
    
    def compute_surrogate_obj(self, state, action, advantage, old_log_prob, actor):
        mu, std = actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        log_prob = action_dist.log_prob(action)
        ratio = torch.exp(log_prob - old_log_prob)
        return torch.mean(ratio * advantage)
    
    def line_search(self, state, action, advantage, old_log_prob, old_action_dist, max_vec, max_iter=15):
        old_parameters = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(state, action, advantage, old_log_prob, self.actor)
        
        for i in range(max_iter):
            coefficient = self.alpha ** i
            new_parameters = old_parameters + coefficient * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_parameters, new_actor.parameters())
            mu, std = new_actor(state)
            new_action_dist = torch.distributions.Normal(mu, std)
            kl_divergence = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist, new_action_dist))
            new_obj = self.compute_surrogate_obj(state, action, advantage, old_log_prob, new_actor)
            # Update condition: improve objective and satisfy KL constraint
            if new_obj > old_obj and kl_divergence < self.kl_constraint:
                return new_parameters
        return old_parameters
    
    def policy_learn(self, state, action, old_action_dist, old_log_prob, advantage):
        # Update policy network
        surrogate_obj = self.compute_surrogate_obj(state, action, advantage, old_log_prob, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # Solving Hx = g using Conjugate Gradient
        descent_direction = self.conjugate_gradient(obj_grad, state, old_action_dist)
        Hx = self.hessian_matrix_vector_product(state, old_action_dist, descent_direction)
        max_step = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hx) + 1e-8))
        new_parameters = self.line_search(state, action, advantage, old_log_prob, old_action_dist,
                                          max_step * descent_direction)
        torch.nn.utils.convert_parameters.vector_to_parameters(new_parameters, self.actor.parameters())
        
    def update(self, transition_dict):
        state = torch.tensor(transition_dict['state'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dict['action']).view(-1, 1).to(self.device)
        reward = torch.tensor(transition_dict['reward'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dict['next_state'], dtype=torch.float).to(self.device)
        done = torch.tensor(transition_dict['done'], dtype=torch.float).view(-1, 1).to(self.device)
        reward = (reward + 8.0) / 8.0 # Reward scaling for Pendulum-v1
        
        td_target = reward + self.gamma * self.critic(next_state) * (1 - done)
        td_error = td_target - self.critic(state)
        # Using GAE to compute advantage estimates
        # .cpu() is to convert tensor from pytorch to numpy for compute_advantage function
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)
        mu, std = self.actor(state)
        old_action_dist = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_prob = old_action_dist.log_prob(action).detach()
        # Update value network
        critic_loss = torch.mean(F.mse_loss(self.critic(state), td_target.detach()))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        # Update policy network
        self.policy_learn(state, action, old_action_dist, old_log_prob, advantage)
        
num_episodes = 2000
hidden_dim = 128
# gamma = 0.99
# lmbda = 0.95
gamma = 0.9
lmbda = 0.9
value_lr = 2e-3
# kl_constraint = 0.002
# alpha = 0.8
kl_constraint = 0.0005
alpha = 0.5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# env_name = 'CartPole-v1'
env_name = 'Pendulum-v1'
env = gym.make(env_name)
env.reset(seed=0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
action_dim = env.action_space.shape[0]
# agent = TRPO(state_dim, action_dim, hidden_dim, kl_constraint, 
# value_lr, gamma, lmbda, alpha, device)
agent = TRPOContinuous(state_dim, action_dim, hidden_dim, kl_constraint,
             value_lr, gamma, lmbda, alpha, device)
return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

episode_list = list(range(len(return_list)))
plt.plot(episode_list, return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
# plt.title('TRPO on CartPole-v1')
plt.title('TRPO on Pendulum-v1')
plt.show()

mv_return_list = rl_utils.moving_average(return_list, 9)
plt.plot(episode_list, mv_return_list)
plt.xlabel('Episode')
plt.ylabel('Return')
#plt.title('TRPO on CartPole-v1')
plt.title('TRPO on Pendulum-v1')
plt.show()