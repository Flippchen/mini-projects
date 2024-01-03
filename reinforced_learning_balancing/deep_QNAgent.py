import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory


def create_model(states, actions):
    model = Sequential(
        [
            Flatten(input_shape=(1, states)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(actions, activation='linear')
        ]
    )
    model.summary()
    return model


# tf.compat.v1.disable_eager_execution() activate if needed
env = gym.make('CartPole-v1')

states = env.observation_space.shape[0]
actions = env.action_space.n
