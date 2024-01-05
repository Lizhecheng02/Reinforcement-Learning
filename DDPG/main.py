from ddpg_pytorch import Agent
import gym
import numpy as np
from utils import plotLearning

env = gym.make("LunarLanderContinuous-v2")
agent = Agent(
    alpha=0.000025,
    beta=0.00025,
    input_dims=[8],
    tau=0.001,
    env=env,
    batch_size=64,
    layer1_size=400,
    layer2_size=300,
    n_actions=2
)

np.random.seed(0)

score_history = []
for i in range(1000):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        new_state, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, new_state, int(done))
        agent.learn()
        score += reward
        observation = new_state
    score_history.append(score)

    print("episode ", i, "score %.2f" % score,
          "trailing 100 games avg %.3f" % np.mean(score_history[-100:]))

filename = "plot.png"
plotLearning(score_history, filename, window=100)
