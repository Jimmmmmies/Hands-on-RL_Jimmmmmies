import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from configs import CliffWalkingEnv, print_agent

class Q_learning:
    
    def __init__(self, ncol, nrow, epsilon, gamma, alpha, n_action=4):
        self.Q = np.zeros((nrow * ncol, n_action))
        self.n_action = n_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q[state])
        return action
    
    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * np.max(self.Q[s1]) - self.Q[s0, a0]
        self.Q[s0, a0] += self.alpha * td_error
        
np.random.seed(0)
ncol = 12
nrow = 4
n_action = 4
epsilon = 0.1
alpha = 0.1
gamma = 0.9
env = CliffWalkingEnv(nrow, ncol)
agent = Q_learning(ncol, nrow, epsilon, alpha, gamma)
n_episodes = 500
action_meaning = ['^', 'v', '<', '>']

return_list = []
for i in range(10):
    with tqdm(total=int(n_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(n_episodes / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while True:
                action = agent.choose_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward
                agent.update(state, action, reward, next_state)
                state = next_state
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
plt.title('Q-learning on Cliff Walking')
plt.show()
print_agent(agent, env, action_meaning, list(range(37, 47)), [47])