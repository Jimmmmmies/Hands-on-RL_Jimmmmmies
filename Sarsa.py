import numpy as np
import matplotlib.pyplot as plt
from configs import CliffWalkingEnv, print_agent
from tqdm import tqdm

class Sarsa:
    
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q = np.zeros((nrow * ncol, n_action))
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q[s1, a1] - self.Q[s0, a0]
        self.Q[s0, a0] += self.alpha * td_error
        
class n_step_Sarsa:
    
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_step=5, n_action=4):
        self.Q = np.zeros((nrow * ncol, n_action))
        self.n_action = n_action
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n_step
        self.state_list = []
        self.action_list = []
        self.reward_list = []
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:
            G = self.Q[s1, a1]
            for i in reversed(range(self.n)):
                G = self.reward_list[i] + self.gamma * G
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q[s, a] += self.alpha * (G - self.Q[s, a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q[s, a] += self.alpha * (G - self.Q[s, a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []

np.random.seed(0)
ncol = 12
nrow = 4
n_step = 5
n_action = 4
epsilon = 0.1
alpha = 0.1
gamma = 0.9
env = CliffWalkingEnv(nrow, ncol)
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
agent1 = n_step_Sarsa(ncol, nrow, epsilon, alpha, gamma)
n_episodes = 500
action_meaning = ['^', 'v', '<', '>']

return_list = []
for i in range(10):
    with tqdm(total=int(n_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(n_episodes / 10)):
            episode_return = 0
            state = env.reset()
            # action = agent.choose_action(state)
            action = agent1.choose_action(state)
            done = False
            while True:
                next_state, reward, done = env.step(action)
                # next_action = agent.choose_action(next_state)
                next_action = agent1.choose_action(next_state)
                episode_return += reward
                # agent.update(state, action, reward, next_state, next_action)
                agent1.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                if done:
                    break
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
plt.title('Sarsa on Cliff Walking')
plt.show()
print_agent(agent1, env, action_meaning, list(range(37, 47)), [47])