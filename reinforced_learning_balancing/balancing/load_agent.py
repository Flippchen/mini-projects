import gym
import numpy as np

from deep_QNAgent import create_model, create_agent
from tensorflow.keras.optimizers import Adam

env = gym.make('CartPole-v1')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = create_model(states, actions)
agent = create_agent(model, actions)

agent.compile(Adam(lr=1e-3), metrics=['mae'])
agent.load_weights('agent_weights.h5f')

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

env.close()
