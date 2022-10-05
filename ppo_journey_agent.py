import acme
import numpy as np
import dm_env
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, Activation, Input
from keras import Sequential, Model
from keras.optimizer_v2.adam import Adam
import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras.losses as kls

class actor(tf.keras.Model):
  def __init__(self, out):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(50,activation='relu')
    # self.d2 = tf.keras.layers.Dense(50,activation='relu')
    self.out = tf.keras.layers.Dense(out,activation='softmax')

  def call(self, input_data, **kwargs):
    x = tf.convert_to_tensor(input_data)
    x = self.d1(x)
    # x = self.d2(x)
    x = self.out(x)
    return x

class critic(tf.keras.Model):
  def __init__(self, out):
    super().__init__()
    self.d1 = tf.keras.layers.Dense(50,activation='relu')
    # self.d2 = tf.keras.layers.Dense(50,activation='relu')
    self.out = tf.keras.layers.Dense(1,activation='softmax')

  def call(self, input_data, **kwargs):
    x = tf.convert_to_tensor(input_data)
    x = self.d1(x)
    # x = self.d2(x)
    x = self.out(x)
    return x

class PPOJourneyAgent():

    def __init__(self, action_space, observation_space):

        # self.model = self.create_model(observation_space, action_space)
        self.actor=actor(action_space)
        self.critic = critic(action_space)
        self.action_space = action_space
        # store timestep, action, next_timestep
        self.state = None
        self.action = None
        self.next_state = None
        self.total_reward = 0
        self.rewards = []
        self.actions = []
        self.states = []
        self.probs = []
        self.dones = []
        self.values = []

        self.gamma = 0.99
        self.act_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.crit_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.complete = False
        self.clip_param = 0.2

    def learn(self, states, actions, adv, old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),-1))
        old_probs = tf.reshape(old_probs, (len(old_probs), self.action_space))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)

        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.act_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.crit_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

    def actor_loss(self, probs, actions, adv, old_probs, closs):

        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []

        for pb, t, op, a in zip(probability, adv, old_probs, actions):
            t = tf.constant(t)
            # op =  tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb[a], op[a])
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # print(loss)
        return loss

    def select_action(self, observation, deployment = False):
        self.states.append(observation)
        prob = self.actor(np.array([observation]))
        self.probs.append(prob[0])
        value = self.critic(np.array([observation]))
        self.values.append(value[0][0])
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def observe_first(self, observation):
        self.state = observation
        self.total_reward = 0
        self.rewards = []
        self.actions = []
        self.states = []
        self.probs = []
        self.dones = []
        self.values = []

    def observe(self, action, reward, observation, last):
        self.action = action
        self.next_state = observation
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_reward+=reward
        self.complete=last
        self.dones.append(1-last)

    def update(self, deployment=False):
        # if self.complete:
        #     value= self.critic(np.array([self.next_state]))
        #     self.values.append(value)
        #     np.reshape(self.probs, (len(self.probs), self.action_space))
        #     probs = np.stack(self.probs, axis=0)
        #     states, actions, returns, adv = self.preprocess1(self.states, self.actions, self.rewards, self.dones,
        #                                                      self.values, 1)
        #     for epochs in range(10):
        #         al, cl = self.learn(states, actions, adv, probs, returns)
        #     print("total reward is {}".format(self.total_reward))
        # else:
        #     self.state = self.next_state

        value= self.critic(np.array([self.next_state]))
        v = self.values.copy()
        v.append(value)
        np.reshape(self.probs, (len(self.probs), self.action_space))
        probs = np.stack(self.probs, axis=0)
        states, actions, returns, adv = self.preprocess1(self.states, self.actions, self.rewards, self.dones,
                                                         v, 1)
        for epochs in range(10):
            al, cl = self.learn(states, actions, adv, probs, returns)
        if self.complete:
            print("total reward is {}".format(self.total_reward))
        self.state = self.next_state

    def preprocess1(self, states, actions, rewards, done, values, gamma):
        g = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            g = delta + gamma * lmbda * self.dones[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - np.array(values[:-1])
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = np.array(returns, dtype=np.float32)
        return states, actions, returns, adv