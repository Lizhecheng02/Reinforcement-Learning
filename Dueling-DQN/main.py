import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn import Agent
from utils import plotlearning

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    num_games = 1000
    load_checkpoint = False

    agent = Agent(
        gamma=0.9,
        epsilon=1.0,
        lr=5e-4,
        input_dims=[8],
        n_actions=4,
        mem_size=100000,
        eps_min=0.01,
        batch_size=64,
        eps_dec=1e-3,
        replace=100
    )

    if load_checkpoint:
        agent.load_models()

    filename = "plot.png"
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        observation = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
        print('episode: ', i, 'score %.1f ' % score, ' average score %.1f' %
              avg_score, 'epsilon %.2f' % agent.epsilon)

        eps_history.append(agent.epsilon)

    x = [i + 1 for i in range(num_games)]
    plotlearning(x=x, scores=scores, epsilons=eps_history, filename=filename)
