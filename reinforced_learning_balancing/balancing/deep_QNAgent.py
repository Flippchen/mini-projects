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
            Flatten(input_shape=(1, )+ states),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(actions, activation='linear')
        ]
    )
    model.summary()
    return model


def create_agent(model, actions):
    policy = BoltzmannGumbelQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2
    )
    return dqn


if "__main__" == __name__:
    # Activate if needed
    tf.compat.v1.disable_eager_execution()
    env = gym.make('CarRacing-v2', continuous=False)

    states = env.observation_space.shape
    actions = env.action_space.n

    model = create_model(states, actions)
    agent = create_agent(model, actions)

    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    agent.fit(env, nb_steps=10000, visualize=False, verbose=1)

    results = agent.test(env, nb_episodes=10, visualize=True)
    print(np.mean(results.history["episode_reward"]))

    agent.save_weights('agent_weights.h5f', overwrite=True)

    env.close()
