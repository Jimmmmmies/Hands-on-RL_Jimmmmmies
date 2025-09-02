import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# 5 7 11 12
# 左 下 右 上
# prob next_s reward done
class MonteCarlo_epsilon_greedy:
    
    def __init__(self):
        self.env = gym.make("FrozenLake-v1")
        self.P = self.env.unwrapped.P
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.N = np.zeros((self.n_states, self.n_actions))
        self.policy = np.zeros(self.n_states)
        self.V = np.zeros(self.n_states)
        self.holes = [5, 7, 11, 12]
        self.end = [15]

    def epsilon_greedy(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return np.argmax([np.random.randint(self.n_actions)])
        else:
            return np.argmax(self.Q[state])

    def sample_episode(self, episode_maxlength, epsilon=0.1):
        episode = []
        episode_len = 0
        state, _ = self.env.reset()
        while episode_len < episode_maxlength:
            action = self.epsilon_greedy(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward, next_state))
            if terminated or truncated:
                break
            episode_len += 1
            state = next_state
        return episode

    def update_policy(self, episodes_num, episode_maxlength, gamma=0.9, epsilon=0.1):
        for _ in range(episodes_num):
            if epsilon > 0.01:
                epsilon -= 0.001
            episode = self.sample_episode(episode_maxlength, epsilon)
            G = 0
            for i in range(len(episode) - 1, -1, -1):
                state, action, reward, next_state = episode[i]
                G = reward + gamma * G
                self.N[state, action] += 1
                self.Q[state, action] += (G - self.Q[state, action]) / self.N[state, action]
        for s in range(self.n_states):
            best_action = np.argmax(self.Q[s])
            self.policy[s] = best_action
            for a in range(self.n_actions):
                if a == best_action:
                    self.V[s] += (1 - (self.n_actions - 1) * epsilon / self.n_actions) * self.Q[s][a]
                else:
                    self.V[s] += epsilon / self.n_actions * self.Q[s][a]

    def display_policy(self):
        for i in range(4):
            for j in range(4):
                if (i * 4 + j) in self.holes:
                    print("#", end=' ')
                elif self.policy[i * 4 + j] == 0:
                    print('←', end=' ')
                elif self.policy[i * 4 + j] == 1:
                    print('↓', end=' ')
                elif self.policy[i * 4 + j] == 2:
                    print('→', end=' ')
                elif self.policy[i * 4 + j] == 3:
                    print('↑', end=' ')
            print()
        for s in range(self.n_states):
            print(f'State {s}: V = {self.V[s]:.2f}, Q = {self.Q[s]}, Policy = {self.policy[s]}')

if __name__ == '__main__':
    agent = MonteCarlo_epsilon_greedy()
    agent.update_policy(10000, 100)
    agent.display_policy()