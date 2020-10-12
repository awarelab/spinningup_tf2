"""Core functions of the VPG algorithm."""
import gym
import numpy as np
import scipy.signal
import tensorflow as tf


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def distribute_value(value, num_proc):
    """Adjusts training parameters for distributed training.

    In case of distributed training frequencies expressed in global steps have
    to be adjusted to local steps, thus divided by the number of processes.
    """
    return max(value // num_proc, 1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def discount_cumsum(x, discount):
    """Magic from rllab for computing discounted cumulative sums of vectors."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[
           ::-1]


@tf.function
def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
            ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
            2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None,
        layer_norm=False):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()

    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=None))
        if layer_norm:
            model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Activation(activation))

    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=None))
    if layer_norm:
        model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Activation(output_activation))

    return model


def make_actor_discrete(observation_space, action_space, hidden_sizes,
                        activation, layer_norm):
    """Creates actor tf.keras.Model.

    This function can be used only in environments with discrete action space.
    """

    class DiscreteActor(tf.keras.Model):
        """Actor model for discrete action space."""

        def __init__(self, observation_space, action_space, hidden_sizes,
                     activation, layer_norm):
            super().__init__()
            self._act_dim = action_space.n

            obs_input = tf.keras.Input(shape=observation_space.shape)
            actor = mlp(
                hidden_sizes=list(hidden_sizes) + [action_space.n],
                activation=activation,
                layer_norm=layer_norm
            )(obs_input)

            self._network = tf.keras.Model(inputs=obs_input, outputs=actor)

        @tf.function
        def call(self, inputs, training=None, mask=None):
            return tf.nn.log_softmax(self._network(inputs))

        @tf.function
        def action(self, observations):
            return tf.squeeze(tf.random.categorical(self(observations), 1),
                              axis=1)

        @tf.function
        def action_logprob(self, observations, actions):
            return tf.reduce_sum(
                tf.math.multiply(self(observations),
                                 tf.one_hot(tf.cast(actions, tf.int32),
                                            depth=self._act_dim)), axis=-1)

    return DiscreteActor(observation_space, action_space, hidden_sizes,
                         activation, layer_norm)


def make_actor_continuous(action_space, hidden_sizes,
                          activation, layer_norm):
    """Creates actor tf.keras.Model.

    This function can be used only in environments with continuous action space.
    """

    class ContinuousActor(tf.keras.Model):
        """Actor model for continuous action space."""

        def __init__(self, action_space, hidden_sizes,
                     activation, layer_norm):
            super().__init__()
            self._action_dim = action_space.shape

            self._body = mlp(
                hidden_sizes=list(hidden_sizes),
                activation=activation,
                layer_norm=layer_norm
            )

            self._mu = tf.keras.layers.Dense(self._action_dim[0], name='mean')
            self._log_std = tf.Variable(
                initial_value=-0.5 * np.ones(shape=(1,) + self._action_dim,
                                             dtype=np.float32), trainable=True,
                name='log_std_dev')

        @tf.function
        def call(self, inputs, training=None, mask=None):
            x = self._body(inputs)
            mu = self._mu(x)
            log_std = tf.clip_by_value(self._log_std, LOG_STD_MIN, LOG_STD_MAX)

            return mu, log_std

        @tf.function
        def action(self, observations):
            mu, log_std = self(observations)
            std = tf.exp(log_std)
            return mu + tf.random.normal(tf.shape(input=mu)) * std

        @tf.function
        def action_logprob(self, observations, actions):
            mu, log_std = self(observations)
            return gaussian_likelihood(actions, mu, log_std)

    return ContinuousActor(action_space, hidden_sizes,
                           activation, layer_norm)


def make_critic(observation_space, hidden_sizes, activation):
    """Creates critic tf.keras.Model"""
    obs_input = tf.keras.Input(shape=observation_space.shape)

    critic = tf.keras.Sequential([
        mlp(hidden_sizes=list(hidden_sizes) + [1],
            activation=activation),
        tf.keras.layers.Reshape([]),
    ])(obs_input)

    return tf.keras.Model(inputs=obs_input, outputs=critic)


def mlp_actor_critic(observation_space, action_space, hidden_sizes=(64, 32),
                     activation=tf.tanh, layer_norm=False):
    """Creates actor and critic tf.keras.Model-s."""
    actor = None

    # default policy builder depends on action space
    if isinstance(action_space, gym.spaces.Discrete):
        actor = make_actor_discrete(observation_space, action_space,
                                    hidden_sizes,
                                    activation, layer_norm)
    elif isinstance(action_space, gym.spaces.Box):
        actor = make_actor_continuous(action_space,
                                      hidden_sizes,
                                      activation, layer_norm)

    critic = make_critic(observation_space, hidden_sizes, activation)

    return actor, critic
