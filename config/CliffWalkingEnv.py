import numpy as np

class CliffWalkingEnv:

    def __init__(self, nrow, ncol):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0
        self.y = self.nrow - 1
        
    def step(self, action):
        movement = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.x = min(self.ncol - 1, max(0, self.x + movement[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + movement[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    
def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    print('Strategy:')
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                Q_max = np.max(agent.Q[i * env.ncol + j])
                str = ''
                for k in range(len(action_meaning)):
                    str += action_meaning[k] if agent.Q[i * env.ncol + j, k] == Q_max else 'o'
                print(str, end=' ')
        print()