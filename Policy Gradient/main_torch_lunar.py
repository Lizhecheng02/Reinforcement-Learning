import numpy as np
import gym
import matplotlib.pyplot as plt
from utils import plotLearning
from gym import wrappers
from pgm_torch import PolicyGradientAgent

if __name__ == "__main__":
    agent = PolicyGradientAgent(
        alpha=0.001,
        input_dims=[8],
        gamma=0.99,
        n_actions=4,
        layer1_size=128,
        layer2_size=128
    )

    env = gym.make("LunarLander-v2")

    score_history = []
    score = 0

    num_episodes = 2500
    for i in range(num_episodes):
        print("episode:", i, "score:", score)
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            observation = observation_
            score += reward

        score_history.append(score)
        agent.learn()

    filename = "plot.png"
    plotLearning(score_history, filename, window=50)
