"""Core functions of the DDPG algorithm."""

import tensorflow as tf


def mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation=activation)
        for size in hidden_sizes
    ], name)


def layer_norm_mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_sizes[0]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation('tanh'),
        mlp(hidden_sizes[1:], activation)
    ], name)


def mlp_actor_critic(action_space,
                     hidden_sizes=(256, 256),
                     activation='elu'):
    """Creates actor and critic based on tf.keras.Model."""
    act_dim = action_space.shape[0]
    act_limit = action_space.high[0]

    class _MlpActor(tf.keras.Model):
        def __init__(self, hidden_sizes, activation):
            super().__init__()
            self._body = layer_norm_mlp(hidden_sizes, activation, name='actor')
            self._pi = tf.keras.layers.Dense(
                act_dim,
                activation='tanh',
                kernel_initializer=tf.keras.initializers.VarianceScaling(1e-4),
                name='pi'
            )

        def call(self, x):  # pylint: disable=arguments-differ
            pi = self._pi(self._body(x))
            pi *= act_limit
            return pi

    class _MlpCritic(tf.keras.Model):
        def __init__(self, hidden_sizes, activation):
            super().__init__()
            self._body = layer_norm_mlp(hidden_sizes, activation, name='critic')
            self._value = tf.keras.layers.Dense(1, name='value')

        def call(self, x):  # pylint: disable=arguments-differ
            q = self._value(self._body(x))
            q = tf.squeeze(q, axis=1)
            return q

    actor = _MlpActor(hidden_sizes, activation)
    critic = _MlpCritic(hidden_sizes, activation)

    return actor, critic
