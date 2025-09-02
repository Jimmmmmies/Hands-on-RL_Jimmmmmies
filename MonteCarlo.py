import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

class MonteCarlo_epsilon_greedy:
    
    def __init__(self):
        self.env = gym.make("FrozenLake-v1")
        self.P = self.env.unwrapped.P
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.N = np.zeros((self.n_states, self.n_actions))
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states)
        self.holes = [5, 7, 11, 12]
        self.end = [15]

    def epsilon_greedy(self, state, epsilon=0.1):
        best_action = np.argmax(self.Q[state])
        probs = np.ones(self.n_actions) * (epsilon / self.n_actions)
        probs[best_action] += (1.0 - epsilon)
        return np.random.choice(np.arange(self.n_actions), p=probs)

    def sample_episode(self, episode_maxlength, epsilon=0.1):
        episode = []
        episode_len = 0
        state, _ = self.env.reset()
        while episode_len < episode_maxlength:
            action = self.epsilon_greedy(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            if terminated and next_state in self.end:
                custom_reward = 5
            elif terminated:
                custom_reward = -10
            else:
                custom_reward = -1.5
            episode.append((state, action, custom_reward, next_state))
            if terminated or truncated:
                break
            episode_len += 1
            state = next_state
        return episode

    def update_policy(self, episodes_num, episode_maxlength, gamma=0.95, epsilon=0.1):
        min_epsilon = 0.01
        epsilon_decay = (epsilon - min_epsilon) / (episodes_num * 0.8)
        for _ in range(episodes_num):
            epsilon = max(min_epsilon, epsilon - epsilon_decay)
            episode = self.sample_episode(episode_maxlength, epsilon)
            G = 0
            # visited = set()
            for i in range(len(episode) - 1, -1, -1):
                state, action, reward, next_state = episode[i]
                G = reward + gamma * G
                # if (state, action) not in visited:
                    # visited.add((state, action))
                self.N[state, action] += 1
                self.Q[state, action] += (G - self.Q[state, action]) / self.N[state, action]
        for s in range(self.n_states):
            self.V[s] = np.max(self.Q[s])

    def display_policy(self):
        for s in range(self.n_states):
            self.policy[s] = np.argmax(self.Q[s])
        for i in range(4):
            for j in range(4):
                if (i * 4 + j) in self.holes:
                    print("#", end=' ')
                elif (i * 4 + j) == self.end[0]:
                    print("🏁", end=' ')
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
            for a in range(self.n_actions):
                print(self.Q[s, a], end=' ')

if __name__ == '__main__':
    agent = MonteCarlo_epsilon_greedy()
    agent.update_policy(100000, 100)
    agent.display_policy()