"""Core functions of SAC algorithm."""

import numpy as np
import tensorflow as tf


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
        ((value - mu) / (tf.exp(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi)
    )
    return tf.reduce_sum(pre_sum, axis=1)


def apply_squashing_func(mu, pi, logp_pi):
    """Applies adjustment to mean, pi and log prob.

    This formula is a little bit magic. To get an understanding of where it
    comes from, check out the original SAC paper (arXiv 1801.01290) and look
    in appendix C. This is a more numerically-stable equivalent to Eq 21.
    Try deriving it yourself as a (very difficult) exercise. :)
    """
    logp_pi -= tf.reduce_sum(
        2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation=activation)
        for size in hidden_sizes
    ], name)


def layer_norm_mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_sizes[0]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation(tf.nn.tanh),
        mlp(hidden_sizes[1:], activation)
    ], name)


def mlp_actor_critic(observation_space,
                     action_space,
                     hidden_sizes=(256, 256),
                     activation=tf.nn.relu):
    """Creates actor and critic tf.keras.Model-s."""
    obs_dim = observation_space.shape[0]
    act_dim = action_space.shape[0]

    # Make the actor.
    class _MlpActor(tf.keras.Model):
        def __init__(self, action_space, hidden_sizes, activation):
            super().__init__()
            self._action_space = action_space
            self._body = mlp(hidden_sizes, activation, name='actor')
            self._mu = tf.keras.layers.Dense(
                action_space.shape[0], name='mean')
            self._log_std = tf.keras.layers.Dense(
                action_space.shape[0], name='log_std_dev')

        def call(self, x):  # pylint: disable=arguments-differ
            x = self._body(x)
            mu = self._mu(x)
            log_std = self._log_std(x)

            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
            pi = mu + tf.random.normal(tf.shape(input=mu)) * std
            logp_pi = gaussian_likelihood(pi, mu, log_std)

            mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

            # Make sure actions are in correct range
            action_scale = self._action_space.high[0]
            mu *= action_scale
            pi *= action_scale

            return mu, pi, logp_pi
    actor = _MlpActor(action_space, hidden_sizes, activation)

    # Make the critic.
    obs_input = tf.keras.Input(shape=(obs_dim,))
    act_input = tf.keras.Input(shape=(act_dim,))
    concat_input = tf.keras.layers.Concatenate(axis=-1)([obs_input, act_input])
    body = tf.keras.Sequential([
        mlp(hidden_sizes, activation, name='critic'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Reshape([])  # Very important to squeeze values!
    ])
    critic = tf.keras.Model(inputs=[obs_input, act_input],
                            outputs=body(concat_input))

    return actor, critic
