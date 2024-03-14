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
from atari_wrappers import make_atari, wrap_deepmind

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
        if agent.n_frames >= 10000000:
            break

        while not done:
            print(agent.n_frames)
            action = agent.get_action(obs, raw=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if terminated:
            #     reward = -1

            agent.remember(obs, action, reward, next_obs, terminated)

            if agent.n_frames > 10000:
                agent.train_long_memory()


            # env.render()


            done = terminated
            obs = next_obs
            counter+=reward
            absolute_counter += 1
            if done:
                reward_per_game.append(reward)
                writer.add_scalar("Reward/Training", counter, len(reward_per_game))
                if len(reward_per_game) > 50:
                    batch_reward.append(sum(reward_per_game[-50:])/50)

                agent.model.save()
                # do some plotting with the loss and shit

def policy_train(env):
    np.random.seed(42)
    random.seed(42)
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
    return np.swapaxes(raw_image_data, 0, 2)

def play(agent, env, n_episodes):
    reward_per_game = []
    batch_reward = []
    agent.n_frames = 0
    agent.epsilon = 0
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        counter = 0
        absolute_counter = 0


        while not done:
            print(agent.n_frames)
            action = agent.get_action(obs, raw=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            # if terminated:
            #     reward = -1

            done = terminated
            obs = next_obs
            counter += reward
            absolute_counter += 1
            if done:
                reward_per_game.append(reward)
                if len(reward_per_game) > 50:
                    batch_reward.append(sum(reward_per_game[-50:]) / 50)
                # do some plotting with the loss and shit








if __name__ == "__main__":
    # env = gym.make('CartPole-v0', render_mode="human")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = make_atari('PongNoFrameskip-v4', render=True, difficulty=3)
    env = wrap_deepmind(env, frame_stack=True, scale=True)

    # print(image_transform(env.reset()[0]))

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
    # agent = DeepAgent(model=deep_qnet.to(device),
    #                   target_model=target_qnet.to(device),
    #                   memory_len= 25000,
    #                   optimizer= optim.RMSprop(deep_qnet.parameters(), lr=1e-4, momentum=0.95),
    #                   criterion=nn.MSELoss(),
    #                   gamma=0.9,
    #                   batch=64,
    #                   transform_func=identity,
    #                   device= device
    #                   )
    #
    # train(env, agent, 100000000)
    # policy_train(env)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dqcnn = DQCNN((4, 84, 84), env.action_space.n)
    dqcnn.load_state_dict(torch.load("model_folder/model.pth"))
    dqcnn.to(device)

    target_dcqnn = DQCNN((4, 84, 84), env.action_space.n)
    target_dcqnn.to(device)

    agent = DeepAgent(model=dqcnn,
                      target_model=target_dcqnn,
                      memory_len=10000,
                      optimizer=optim.Adam(dqcnn.parameters(), lr=1e-4),# optim.RMSprop(dqcnn.parameters(), lr=0.000025, momentum=0.95)
                      criterion=nn.HuberLoss(),
                      gamma=0.99,
                      batch=32,
                      transform_func=image_transform,
                      device=device
                      )

    play(agent, env, 1000)
    # train(env, agent, 100000000)

