"""Core functions of the DDPG algorithm."""

import tensorflow as tf


def mlp(hidden_sizes=(32,), activation='relu', output_activation=None):
    model = tf.keras.Sequential()

    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))

    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1],
                                    activation=output_activation))
    return model


def mlp_actor_critic(action_space,
                     hidden_sizes=(256, 256),
                     activation='relu'):
    """Creates actor and critic based on tf.keras.Model."""
    act_dim = action_space.shape[0]
    act_limit = action_space.high[0]

    class _MlpActor(tf.keras.Model):
        def __init__(self, hidden_sizes, activation):
            super().__init__()
            self._body = mlp(list(hidden_sizes) + [act_dim], activation)

        def call(self, x):  # pylint: disable=arguments-differ
            x = self._body(x)
            # Make sure actions are in the correct range
            pi = tf.tanh(x) * act_limit
            return pi

    class _MlpCritic(tf.keras.Model):
        def __init__(self, hidden_sizes, activation):
            super().__init__()
            self._body = mlp(list(hidden_sizes) + [1], activation)

        def call(self, x):  # pylint: disable=arguments-differ
            x = self._body(x)
            q = tf.squeeze(x, axis=1)
            return q

    actor = _MlpActor(hidden_sizes, activation)
    critic1 = _MlpCritic(hidden_sizes, activation)
    critic2 = _MlpCritic(hidden_sizes, activation)

    return actor, critic1, critic2
