import gymnasium as gym
import rubiks_cube_gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import DeepAgent, RandomAgent
from model import Deep_QNet, DQCNN
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter()
import random

plt.ion()
def train(env, agent, n_episodes):
    reward_per_game = []
    batch_reward = []
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        counter = 0
        absolute_counter = 0
        agent.n_episodes +=1

        while not done:
            action = agent.get_action(obs, raw=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if done:
            #     reward = -100

            agent.train_short_memory(obs, action, reward, next_obs, terminated)

            agent.remember(obs, action, reward, next_obs, terminated)
            env.render()


            done = terminated
            obs = next_obs
            counter+=reward
            absolute_counter += 1
            if done:
                reward_per_game.append(reward)
                writer.add_scalar("Reward/Training", counter, len(reward_per_game))
                if len(reward_per_game) > 50:
                    batch_reward.append(sum(reward_per_game[-50:])/50)


                agent.train_long_memory()
                agent.model.save()
                # do some plotting with the loss and shit

def policy_train(env):
    num_inputs = 128
    num_actions = env.action_space.n

    model = torch.nn.Sequential(
        torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(128, num_actions, bias=False, dtype=torch.float32),
        torch.nn.Softmax(dim=1)
    )
    epsilon = 0.01
    loss_func = nn.CrossEntropyLoss()

    def run_episode(max_steps_per_episode=10000, render=False):
        states, actions, probs, rewards = [], [], [], []
        state, info = env.reset()
        for _ in range(max_steps_per_episode):
            action_probs = model(torch.tensor(torch.from_numpy(np.expand_dims(state, 0)), dtype=torch.float))[0]
            print(np.squeeze(action_probs.detach().numpy()))
            action = np.random.choice(num_actions, p=np.squeeze(action_probs.detach().numpy()))
            if random.random() > epsilon:
                action = random.randint(0,num_actions- 1)
            print(action)
            nstate, reward, done, terminated, info = env.step(action)
            env.render()
            if done or terminated:
                break

            states.append(state)
            actions.append(action)
            probs.append(action_probs.detach().numpy())
            rewards.append(reward)
            state = nstate

        return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)


    eps = 0.0001

    def discounted_rewards(rewards, gamma=0.99, normalize=True):
        ret = []
        s = 0
        for r in rewards[::-1]:
            s = r + gamma * s
            ret.insert(0, s)
        if normalize:
            ret = (ret - np.mean(ret)) / (np.std(ret) + eps)
        return ret

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    def train_on_batch(x, y):
        x = torch.tensor( torch.from_numpy(x),dtype=torch.float)
        y = torch.tensor( torch.from_numpy(y), dtype=torch.float)
        optimizer.zero_grad()
        predictions = model(x)
        loss = loss_func(predictions, y)
        loss.backward()
        optimizer.step()
        return loss

    alpha = 1e-4

    history = []
    for epoch in range(300):
        states, actions, probs, rewards = run_episode(render=True)
        print(states.dtype)
        one_hot_actions = np.eye(num_actions)[actions.T][0]
        gradients = one_hot_actions - probs
        dr = discounted_rewards(rewards, normalize=True)
        gradients *= dr
        target = alpha * np.vstack([gradients]) + probs
        train_on_batch(states, target)
        writer.add_scalar("Reward/Training",np.sum(rewards), len(history))
        history.append(np.sum(rewards))
        if epoch % 100 == 0:
            print(f"{epoch} -> {np.sum(rewards)}")

def image_transform(raw_image_data):
    image_array = np.reshape(np.array(raw_image_data), (210, 160, 3))
    image_array = np.mean(image_array, axis=2)

    return np.expand_dims(image_array/np.max(image_array), axis=0)








if __name__ == "__main__":
    env = gym.make("Pong-v0", render_mode="human")

    print(image_transform(env.reset()[0]))
    # agent = RandomAgent(env.action_space.n)
    # deep_qnet = Deep_QNet(input_size=len(env.reset()[0]),
    #                       hidden_size=128,
    #                       output_size=env.action_space.n)
    #
    # target_qnet= Deep_QNet(input_size=len(env.reset()[0]),
    #                       hidden_size=128,
    #                       output_size=env.action_space.n)
    # def identity(x):
    #     return x
    # agent = DeepAgent(model=deep_qnet,
    #                   target_model=target_qnet,
    #                   memory_len= 25000,
    #                   optimizer= optim.Adam(deep_qnet.parameters(), lr=5e-5),
    #                   criterion=nn.MSELoss(),
    #                   gamma=0.9,
    #                   batch=64,
    #                   transform_func=identity,
    #                   )
    #
    # train(env, agent, 100000000)
    # policy_train(env)

    dqcnn = DQCNN((1, 210, 160), env.action_space.n)

    target_dcqnn = DQCNN((1, 210, 160), env.action_space.n)

    agent = DeepAgent(model=dqcnn,
                      target_model=target_dcqnn,
                      memory_len= 25000,
                      optimizer= optim.Adam(dqcnn.parameters(), lr=5e-5),
                      criterion=nn.MSELoss(),
                      gamma=0.9,
                      batch=64,
                      transform_func=image_transform,
                      )

    train(env, agent, 100000000)

