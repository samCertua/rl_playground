import acme
import numpy as np
import dm_env
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, Activation, Input
from keras import Sequential, Model
from keras.optimizer_v2.adam import Adam
import tensorflow_probability as tfp
import tensorflow as tf

class model(tf.keras.Model):
  def __init__(self, out):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(10,activation='relu')
    # self.d2 = tf.keras.layers.Dense(50,activation='relu')
    self.out = tf.keras.layers.Dense(out,activation='softmax')

  def call(self, input_data, **kwargs):
    x = tf.convert_to_tensor(input_data)
    x = self.d1(x)
    # x = self.d2(x)
    x = self.out(x)
    return x

class MonteCarlo(acme.Actor):

    def __init__(self, env: dm_env):

        action_space = env.action_spec()
        observation_space = env.observation_spec()
        # self.model = self.create_model(observation_space, action_space)
        self.model=model(action_space.num_values)

        # store timestep, action, next_timestep
        self.timestep = None
        self.action = None
        self.next_timestep = None
        self.total_reward = 0
        self.rewards = []
        self.actions = []
        self.states = []
        self.gamma = 0.99
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.001)


    def create_model(self, observation_space, action_space):
        input = Input(shape=(observation_space.shape))
        flat = Flatten()(input)
        x = Dense(10, activation="relu")(flat)
        out = Dense(action_space.num_values, activation="softmax")(x)
        model = Model(inputs=input, outputs=out)
        # model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def train(self, states, rewards, actions):
        sum_reward = 0
        discnt_rewards = []
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()

        for state, reward, action in zip(states, discnt_rewards, actions):
            with tf.GradientTape() as tape:
                p = self.model(np.array([state]), training=True)
                loss = self.a_loss(p, action, reward)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def a_loss(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss

    def select_action(self, observation, deployment = False):
        # if isinstance(observation, tuple):
        #     observation = observation[0]
        self.states.append(observation)
        prob = self.model(np.array([observation]))
        if deployment:
            action = tf.convert_to_tensor([tf.math.argmax(prob[0])])
        else:
            dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
            action = dist.sample()
        return int(action.numpy()[0])

    def observe_first(self, timestep):
        self.timestep = timestep
        self.total_reward = 0
        self.rewards = []
        self.actions = []
        self.states = []

    def observe(self, action, next_timestep):
        self.action = action
        self.next_timestep = next_timestep
        self.actions.append(action)
        self.rewards.append(next_timestep.reward)
        self.total_reward+=next_timestep.reward

    def update(self, deployment=False):
        if self.next_timestep.last():
            # print(self.actions)
            # print(self.states)
            if not deployment:
                self.train(self.states, self.rewards, self.actions)
            print("total reward is {}".format(self.total_reward))
        else:
            self.timestep = self.next_timestep