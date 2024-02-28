import random
import numpy as np
import random
from collections import deque
import torch



class DeepAgent:
    def __init__(self, model, memory_len, optimizer, criterion, gamma, lr, batch, n_moves):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=memory_len)
        self.n_moves = n_moves
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch

    def get_state(self, raw_state):
        pass

    def remember(self, raw_state, action, reward, next_state, done):
        self.memory.append((self.get_state(raw_state), action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) >self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)

        else:
            mini_sample = self.memory

        states, raw_actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, raw_actions, rewards, next_states, dones)



    def train_short_memory(self, raw_state, raw_action, reward, next_state, done):
        self.train_step(raw_state, raw_action, reward, next_state, done)

    def get_action(self, raw_state, raw=True):
        state = self.get_state(raw_state)
        # epsilon-greedy policy

        self.epsilon = 0.20
        final_action = [0 for i in range(self.n_moves)]
        if random.random() < self.epsilon:
            final_action[random.randint(0, self.n_moves-1)] = 1

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_index = torch.argmax(prediction).item()
            final_action[move_index] = 1

        if not raw:
            return final_action
        else:
            return final_action.index(1)


    def train_step(self, raw_state, raw_action, reward, next_state, done):
        state = self.get_state(raw_state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        raw_action = torch.tensor(raw_action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            raw_action = torch.unsqueeze(raw_action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][raw_action] = Q_new
        # 2: reward update function: r + y *max(next_predicted Q value)

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

class RandomAgent:
    def __init__(self, n_moves):
        self.n_moves = n_moves

    def get_state(self, raw_state):
        pass

    def remember(self, raw_state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, raw_state, raw_action, reward, next_state, done):
        pass

    def get_action(self, raw_state, raw=True):
        final_action = [0 for i in range(self.n_moves)]

        final_action[random.randint(0, self.n_moves - 1)] = 1


        if not raw:
            return final_action
        else:
            return final_action.index(1)

    def train_step(self, raw_state, raw_action, reward, next_state, done):
        pass