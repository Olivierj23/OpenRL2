import random
import numpy as np
import random
from collections import deque
import torch
from replay_buffers import ReplayMemory, PrioritizedExperienceReplayBuffer, Experience, ReplayBuffer
from replay_buffer import PrioritizedReplayBuffer
from model import Deep_QNet
import time


class DeepAgent:
    def __init__(self, model, target_model, memory_len, optimizer, criterion, gamma, batch, transform_func, device):
        self.n_episodes = 0
        self.big_frames = 2000000
        self.epsilon = 1
        self.beta_decay = 0.7
        self.gamma = gamma
        # self.memory = ReplayMemory(max_size=memory_len) # deque(maxlen=memory_len)
        # self.memory = PrioritizedExperienceReplayBuffer(batch_size=batch, buffer_size=memory_len, alpha=0.4)
        #self.memory = ReplayBuffer(memory_len, alpha=0.7)
        self.memory = PrioritizedReplayBuffer(size=memory_len, alpha=0.7)
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
        # self.memory.append((self.get_state(raw_state), action, reward, self.get_state(next_state), done))
        # experience = Experience(self.get_state(raw_state), action, reward, self.get_state(next_state), done)
        self.memory.add(self.get_state(raw_state), action, reward, self.get_state(next_state), done)
        # self.memory.add(experience)
        if done:
            self.n_episodes += 1
        else:
            self.n_frames += 1

    def exponential_annealing_schedule(self, n, rate):
        return 1 - np.exp(-rate * n)


    def train_long_memory(self):
        r = max((self.big_frames - self.n_frames)/self.big_frames, 0)
        self.epsilon = (1-0.1)*r + 0.1

        if len(self.memory) > self.batch_size:
            beta = self.exponential_annealing_schedule(self.n_episodes, self.beta_decay)
            states, raw_actions, rewards, next_states, dones, weights, idxs = self.memory.sample(self.batch_size, beta)
            # idxs, mini_sample, weights = self.memory.sample(beta)
            # idxs, mini_sample, weights = self.memory.sample(self.batch_size, beta)
            # states, raw_actions, rewards, next_states, dones = zip(*mini_sample)
            # print(mini_sample["obs"].shape)
            # states, raw_actions, rewards, next_states, dones = mini_sample["obs"], mini_sample["action"], mini_sample["reward"], \
            #      mini_sample["next_obs"], mini_sample["done"]


            self.train_step(np.array(states), np.array(raw_actions), np.array(rewards), np.array(next_states),
                            np.array(dones), idxs,  np.array(weights))

        else:
            pass




    def train_short_memory(self, raw_state, raw_action, reward, next_state, done, idxs, weights):
        self.train_step(self.get_state(raw_state), raw_action, reward, self.get_state(next_state), done, idxs, weights)

    def get_action(self, raw_state, raw=True):
        if self.n_frames % 1000:
            self.target_model.load_state_dict(self.model.state_dict())

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





    def train_step(self, state, raw_action, reward, next_state, done, idxs, weights):

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        raw_action = torch.tensor(raw_action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        done = torch.tensor(done, dtype=torch.float).to(self.device)
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

        deltas = target_q_values - input_q_values
        priorities = (deltas.abs()
                      .cpu()
                      .detach()
                      .numpy()
                      .flatten())


        self.memory.update_priorities(idxs, priorities + 1e-6)
        weights = torch.tensor(weights, device=self.device)

        loss = torch.mean(self.criterion(input_q_values, target_q_values) * weights)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

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