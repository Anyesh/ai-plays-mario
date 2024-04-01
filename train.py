from collections import deque

import gym_super_mario_bros

from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from utils import preprocess_state
from model import DQNAgent

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, RIGHT_ONLY)

action_space = env.action_space.n
state_space = (80, 88, 1)
num_episodes = 1000000
num_timesteps = 400000
batch_size = 64
DEBUG_LENGTH = 300

dqn = DQNAgent(state_space, action_space)
stuck_buffer = deque(maxlen=DEBUG_LENGTH)


for i in range(num_episodes):
    Return = 0
    done = False
    time_step = 0
    onGround = 79

    state = preprocess_state(env.reset())
    state = state.reshape(-1, 80, 88, 1)

    for t in range(num_timesteps):
        env.render()
        time_step += 1

        if t > 1 and stuck_buffer.count(stuck_buffer[-1]) > DEBUG_LENGTH - 50:
            action = dqn.act(state, onGround=79)
        else:
            action = dqn.act(state, onGround)

        print(f"ACTION IS{str(action)}")

        next_state, reward, done, info = env.step(action)

        onGround = info["y_pos"]
        stuck_buffer.append(info["x_pos"])

        next_state = preprocess_state(next_state)
        next_state = next_state.reshape(-1, 80, 88, 1)

        dqn.store_transition(state, action, reward, next_state, done)
        state = next_state

        Return += reward
        print(
            "Episode is: {}\nTotal Time Step: {}\nCurrent Reward: {}\nEpsilon is: {}".format(
                str(i), str(time_step), str(Return), str(dqn.epsilon)
            )
        )
        if done:
            break

        if len(dqn.memory) > batch_size and i > 0:
            dqn.train(batch_size)

    dqn.update_epsilon(i)
    dqn.update_target_network()
    dqn.save("artifacts/model.h5")

env.close()
