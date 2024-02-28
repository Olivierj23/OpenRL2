import gymnasium as gym
import rubiks_cube_gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from agents import DeepAgent, RandomAgent
from IPython import display


def train(env, agent, n_episodes):
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs, raw=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.train_short_memory(obs, action, reward, next_obs, terminated)

            agent.remember(obs, action, reward, next_obs, terminated)
            env.render()


            done = terminated or truncated
            obs = next_obs
            if done:
                agent.train_long_memory()
                # agent.model.save()
                # do some plotting with the loss and shit
                break

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = RandomAgent(env.action_space.n)

    train(env, agent, 100000000)