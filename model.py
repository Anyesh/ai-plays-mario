from collections import deque
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        # Create variables for our agent
        self.state_space = state_size
        self.action_space = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.8
        self.chosenAction = 0

        # Exploration vs explotation
        self.epsilon = 0.1
        self.max_epsilon = 1
        self.min_epsilon = 0.01
        self.decay_epsilon = 0.0001

        # Building Neural Networks for Agent
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.update_target_network()

    def build_network(self):
        model = Sequential()
        model.add(
            Conv2D(64, (4, 4), strides=4, padding="same", input_shape=self.state_space)
        )
        model.add(Activation("relu"))

        model.add(Conv2D(64, (4, 4), strides=2, padding="same"))
        model.add(Activation("relu"))

        model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())

        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))

        model.compile(loss="mse", optimizer=Adam())

        return model

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def act(self, state, onGround):
        if onGround < 83:
            print("On Ground")
            if random.uniform(0, 1) < self.epsilon:
                self.chosenAction = np.random.randint(self.action_space)
                return self.chosenAction
            Q_value = self.main_network.predict(state)
            self.chosenAction = np.argmax(Q_value[0])
        else:
            print("Not on Ground")

        return self.chosenAction

    def update_epsilon(self, episode):
        self.epsilon = self.min_epsilon + (
            self.max_epsilon - self.min_epsilon
        ) * np.exp(-self.decay_epsilon * episode)

    # train the network
    def train(self, batch_size):
        # minibatch from memory
        minibatch = random.sample(self.memory, batch_size)

        # Get variables from batch so we can find q-value
        for state, action, reward, next_state, done in minibatch:
            target = self.main_network.predict(state)
            print(target)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(
                    self.target_network.predict(next_state)
                )

            self.main_network.fit(state, target, epochs=1, verbose=0)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_pred_act(self, state):
        Q_values = self.main_network.predict(state)
        print(Q_values)
        return np.argmax(Q_values[0])

    def load(self, name):
        self.main_network = load_model(name)
        self.target_network = load_model(name)

    def save(self, name):
        save_model(self.main_network, name)
