import random
import numpy as np
import random
from collections import deque
import torch
from model import Deep_QNet



class DeepAgent:
    def __init__(self, model, target_model, memory_len, optimizer, criterion, gamma, batch, transform_func):
        self.n_episodes = 0
        self.big_n = 10000
        self.epsilon = 1
        self.epsilon_decay = 0.99998
        self.gamma = gamma
        self.memory = deque(maxlen=memory_len)
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch
        self.transform_func = transform_func
        self.target_model.load_state_dict(self.model.state_dict())

    def get_state(self, raw_state):
        return self.transform_func(raw_state)

    def remember(self, raw_state, action, reward, next_state, done):
        self.memory.append((self.get_state(raw_state), action, reward, self.get_state(next_state), done))

    def train_long_memory(self):
        # r = max((self.big_n - self.n_episodes)/self.big_n, 0)
        # self.epsilon = (1-0.01)*r + 0.01
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
        if len(self.memory) >self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)

        else:
            mini_sample = self.memory

        states, raw_actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, raw_actions, rewards, next_states, dones)

        if self.n_episodes % 10:
            self.target_model.load_state_dict(self.model.state_dict())


    def train_short_memory(self, raw_state, raw_action, reward, next_state, done):
        self.train_step(raw_state, raw_action, reward, next_state, done)

    def get_action(self, raw_state, raw=True):
        state = self.get_state(raw_state)
        # epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randrange(self.model.output_size)

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            state0 = torch.unsqueeze(state0, 0)
            print(state0.shape)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()



    def train_step(self, raw_state, raw_action, reward, next_state, done):
        state = self.get_state(raw_state)
        next_state = self.get_state(next_state)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        raw_action = torch.tensor(raw_action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        done = torch.tensor(done, dtype=torch.float)
        done = torch.unsqueeze(done, 0)



        if len(done) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            raw_action = torch.unsqueeze(raw_action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = done

        # 1: predicted Q values with current state
        # pred = self.model(state)
        # target = pred.clone()
        # for idx in range(len(done)):
        #     Q_new = reward[idx]
        #     if not done[idx]:
        #         Q_new = reward[idx] + (self.gamma * torch.max(self.target_model(next_state[idx])))
        #
        #
        #     target[idx][raw_action] = Q_new

        Q_max_action = self.model(next_state).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.target_model(next_state).gather(1, Q_max_action).reshape(-1)

        # Compute the expected Q values
        Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))

        Q_expected = self.model(state).gather(1, raw_action.unsqueeze(1))
        # 2: reward update function: r + y *max(next_predicted Q value)



        self.optimizer.zero_grad()
        # print(target, pred)
        # loss = self.criterion(target, pred)

        # print(Q_expected)
        # print(Q_targets.unsqueeze(1))
        loss = self.criterion(Q_expected, Q_targets.unsqueeze(1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)

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