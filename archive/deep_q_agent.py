import acme
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, Activation, Input
from keras import Sequential, Model
from keras.optimizer_v2.adam import Adam
from collections import deque
import random

DISCOUNT = 0.8
REPLAY_MEMORY_SIZE = 500  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 300  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 300  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)

# epsilon greedy policy
def epsilon_greedy(q_values, epsilon):
    if epsilon < np.random.random():
        return np.argmax(q_values)
    else:
        return np.random.choice(len(q_values))


class DeepQLearningAgent(acme.Actor):

    def __init__(self, env_specs=None, step_size=0.1, q=(10, 10), epsilon=True, init_mode="zeros"):

        # Main model
        self.model = self.create_model(q)
        q = list(q)
        self.action_space = q[-1]
        self.observation_space = q[:-1]
        # Target network
        self.target_model = self.create_model(q)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # set step size
        self.step_size = step_size

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.epsilon = 0.1
        self.total_reward=0

        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None

    def create_conv_model(self, q):
        q = list(q)
        action_space = q[-1]
        observation_space = q[:-1]
        model = Sequential()

        model = Conv2D(256, (3, 3),
                         input_shape=observation_space)(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.2)(model)

        # model = Conv2D(256, (3, 3))(model)
        # model = Activation('relu')(model)
        # model = MaxPooling2D(pool_size=(2, 2))(model)
        # model = Dropout(0.2)(model)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        model.add(Dense(action_space, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def create_model(self, q):
        init = tf.keras.initializers.HeUniform()
        q = list(q)
        action_space = q[-1]
        observation_space = q[:-1]
        input = Input(shape=((4),))
        # flat = Flatten()(input)
        x = Dense(10, kernel_initializer=init)(input)
        out = Dense(action_space, activation="softmax")(x)
        model = Model(inputs = input, outputs = out)
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def select_action(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        if np.random.random() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.get_qs(observation))
        else:
            # print("Epsilon pred")
            # Get random action
            action = np.random.randint(0, self.action_space)
        return action

    def observe_first(self, timestep):
        self.total_reward = 0
        self.timestep = timestep

    def observe(self, action, next_timestep):
        self.action = action
        self.next_timestep = next_timestep
        self.total_reward+=next_timestep.reward

    def get_qs(self, state):
        # state = [state]
        return self.model.predict(state.reshape(1,-1)).reshape(-1, self.action_space)[0]

    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = [transition[0] for transition in minibatch]
        current_states = np.asarray([i[0] if isinstance(i,tuple)  else i for i in current_states])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        # future_qs_list = self.target_model.predict(new_current_states)
        future_qs_list = self.model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state) in enumerate(minibatch):
            current_state = current_state[0] if isinstance(current_state, tuple) else current_state
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not self.next_timestep.last():
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=1, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def update(self):
        # get variables for convenience
        state = self.timestep.observation
        _, reward, discount, next_state = self.next_timestep
        action = self.action
        # print("state: " + str(state) + " action: " + str(action) + " reward: " + str(reward))
        self.update_replay_memory((state, action, reward, next_state))
        self.train(self.timestep.last())
        # if td_error>0 and action!=1:
        #     f =0
        #     pass

        # finally, set timestep to next_timestep
        self.timestep = self.next_timestep


