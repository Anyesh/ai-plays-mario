import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from utils import preprocess_state

from model import DQNAgent

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)

action_space = env.action_space.n
state_space = (80, 88, 1)


dqn = DQNAgent(state_space, action_space)

dqn.load("artifacts/model.h5")


total_reward = 0
while True:
    done = False
    state = preprocess_state(env.reset())
    state = state.reshape(-1, 80, 88, 1)
    onGround = 79

    while not done:
        env.render()
        action = dqn.act(state, onGround)
        next_state, reward, done, info = env.step(action)

        onGround = info["y_pos"]

        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1, 80, 88, 1)
        state = next_state

env.close()
