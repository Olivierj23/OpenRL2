import random
import numpy as np
import random
from collections import deque
import torch
from model import Deep_QNet


class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]
class DeepAgent:
    def __init__(self, model, target_model, memory_len, optimizer, criterion, gamma, batch, transform_func, device):
        self.n_episodes = 0
        self.big_frames = 1000000
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.gamma = gamma
        self.memory = ReplayMemory(max_size=memory_len) # deque(maxlen=memory_len)
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch
        self.transform_func = transform_func
        self.target_model.load_state_dict(self.model.state_dict())
        self.device = device
        self.n_frames = 0

    def get_state(self, raw_state):
        return self.transform_func(raw_state)

    def remember(self, raw_state, action, reward, next_state, done):
        self.memory.append((self.get_state(raw_state), action, reward, self.get_state(next_state), done))

    def train_long_memory(self):
        r = max((self.big_frames - self.n_frames)/self.big_frames, 0)
        self.epsilon = (1-0.1)*r + 0.1

        # if self.epsilon > 0.1:
        #     self.epsilon *= self.epsilon_decay
        if self.memory.size > self.batch_size:
            mini_sample = self.memory.sample(self.batch_size)


        else:
            mini_sample = self.memory.sample(self.memory.size)

        states, raw_actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(np.array(states), np.array(raw_actions), np.array(rewards), np.array(next_states), np.array(dones))




    def train_short_memory(self, raw_state, raw_action, reward, next_state, done):
        self.train_step(self.get_state(raw_state), raw_action, reward, self.get_state(next_state), done)

    def get_action(self, raw_state, raw=True):
        if self.n_frames % 1000:
            self.target_model.load_state_dict(self.model.state_dict())

        self.n_frames += 1
        print(self.epsilon)
        state = self.get_state(raw_state)
        # epsilon-greedy policy
        if random.random() < self.epsilon:
            return random.randrange(self.model.output_size)

        else:
            state0 = torch.tensor(state, dtype=torch.float)
            state0 = state0.to(device=self.device)
            state0 = torch.unsqueeze(state0, 0)
            prediction = self.model(state0)
            print(torch.argmax(prediction).item())
            return torch.argmax(prediction).item()





    def train_step(self, state, raw_action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        state = state.to(self.device)

        next_state = torch.tensor(next_state, dtype=torch.float)
        next_state = next_state.to(self.device)

        raw_action = torch.tensor(raw_action, dtype=torch.long)
        raw_action = raw_action.to(self.device)

        reward = torch.tensor(reward, dtype=torch.float)
        reward = reward.to(self.device)


        done = torch.tensor(done, dtype=torch.float)
        done = done.to(self.device)
        done = torch.unsqueeze(done, 0)



        if len(done.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            raw_action = torch.unsqueeze(raw_action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = done



        # Q_max_action = self.model(next_state).detach().max(1)[1].unsqueeze(1)
        # Q_targets_next = self.target_model(next_state).gather(1, Q_max_action).reshape(-1)
        #
        #
        # # Compute the expected Q values
        # Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))
        #
        # Q_expected = self.model(state).gather(1, raw_action.unsqueeze(1))
        # print(self.model(state))
        # 2: reward update function: r + y *max(next_predicted Q value)

        with torch.no_grad():
            _, max_next_action = self.model(next_state).max(1)
            max_next_q_values = self.target_model(next_state).gather(1, max_next_action.unsqueeze(1)).squeeze()
            target_q_values = reward + (1 - done) * self.gamma * max_next_q_values

        input_q_values = self.model(state)
        input_q_values = input_q_values.gather(1, raw_action.unsqueeze(1)).squeeze()



        self.optimizer.zero_grad()
        # print(target, pred)
        # loss = self.criterion(target, pred)

        print(input_q_values)
        print(target_q_values)
        # loss = self.criterion(Q_expected.T, Q_targets)
        loss = self.criterion(input_q_values, target_q_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

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