"""Core functions of the TD3 algorithm."""

import tensorflow as tf


def mlp(hidden_sizes, activation, trainable=True, name=None):
    """Creates MLP with the specified parameters."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation=activation, trainable=trainable)
        for size in hidden_sizes
    ], name)


class MLPActorCriticFactory:
    """Factory of MLP stochastic actors and critics.

    Args:
        observation_space (gym.spaces.Box): A continuous observation space
          specification.
        action_space (gym.spaces.Box): A continuous action space
          specification.
        hidden_sizes (list): A hidden layers shape specification.
        activation (tf.function): A hidden layers activations specification.
        act_noise (float): Stddev for Gaussian exploration noise.
        ac_number (int): Number of the actor-critic models in the ensemble.
        prior_weight (float): Randomly initialized network output scaling.
    """

    def __init__(self, observation_space, action_space, hidden_sizes,
                 activation, prior_weight, act_noise, ac_number):
        self._obs_dim = observation_space.shape[0]
        self._act_dim = action_space.shape[0]
        self._act_scale = action_space.high[0]
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._act_noise = act_noise
        self._ac_number = ac_number
        self._prior_weight = prior_weight

    def _make_actor(self):
        """Constructs and returns the actor model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        body = mlp(self._hidden_sizes, self._activation)(obs_input)
        mu = tf.keras.layers.Dense(self._act_dim)(body)

        # Normalize the actions.
        g = tf.reduce_mean(tf.math.abs(mu), axis=1, keepdims=True)
        g = tf.maximum(g, tf.ones_like(g))
        mu = mu / g

        # Add the action noise.
        pi = mu + self._act_noise * tf.random.normal(tf.shape(mu))

        # Put the actions in the limit.
        mu = tf.tanh(mu) * self._act_scale
        pi = tf.tanh(pi) * self._act_scale

        return tf.keras.Model(inputs=obs_input, outputs=[mu, pi])

    def make_actor(self):
        """Constructs and returns the ensemble of actor models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        mus, pis = [], []
        for obs_input in tf.unstack(obs_inputs, axis=0):
            model = self._make_actor()
            mu, pi = model(obs_input)
            mus.append(mu)
            pis.append(pi)
        return tf.keras.Model(inputs=obs_inputs, outputs=[
            tf.stack(mus, axis=0),
            tf.stack(pis, axis=0),
        ])

    def _make_critic(self, trainable=True):
        """Constructs and returns the critic model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        act_input = tf.keras.Input(shape=(self._act_dim,))

        concat_input = tf.keras.layers.Concatenate(
            axis=-1)([obs_input, act_input])

        q = tf.keras.Sequential([
            mlp(self._hidden_sizes, self._activation, trainable=trainable),
            tf.keras.layers.Dense(1, trainable=trainable),
            tf.keras.layers.Reshape([])  # Very important to squeeze values!
        ])(concat_input)

        return tf.keras.Model(inputs=[obs_input, act_input], outputs=q)

    def make_critic(self):
        """Constructs and returns the ensemble of critic models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        act_inputs = tf.keras.Input(shape=(None, self._act_dim),
                                    batch_size=self._ac_number)
        qs = []
        for obs_input, act_input in zip(tf.unstack(obs_inputs, axis=0),
                                        tf.unstack(act_inputs, axis=0)):
            model = self._make_critic()
            q = model([obs_input, act_input])

            if self._prior_weight > 0:
                prior = self._make_critic(trainable=False)
                q += self._prior_weight * prior([obs_input, act_input])

            qs.append(q)
        return tf.keras.Model(inputs=[obs_inputs, act_inputs],
                              outputs=tf.stack(qs, axis=0))


class OUProcess(tf.keras.layers.Layer):
    """Layer that adds OU noise to its input."""
    def __init__(self, size, dumping=0.15, stddev=0.2):
        super(OUProcess, self).__init__()
        self._dumping = dumping
        self._stddev = stddev
        self._size = size
        self._noise_value = tf.Variable(tf.zeros(self._size), trainable=False)

    def call(self, inputs, **kwargs):
        del kwargs
        new_noise = (1 - self._dumping) * self._noise_value
        new_noise += tf.random.normal((self._size,)) * self._stddev
        self._noise_value.assign(new_noise)
        return inputs + new_noise


class MLPActorCriticFactoryOUNoise:
    """Factory of MLP stochastic actors and critics with the OU aciton noise.

    Args:
        observation_space (gym.spaces.Box): A continuous observation space
          specification.
        action_space (gym.spaces.Box): A continuous action space
          specification.
        hidden_sizes (list): A hidden layers shape specification.
        activation (tf.function): A hidden layers activations specification.
        act_noise (float): Stddev for Gaussian exploration noise.
        ac_number (int): Number of the actor-critic models in the ensemble.
        prior_weight (float): Randomly initialized network output scaling.
    """

    def __init__(self, observation_space, action_space, hidden_sizes,
                 activation, prior_weight, act_noise, ac_number):
        self._obs_dim = observation_space.shape[0]
        self._act_dim = action_space.shape[0]
        self._act_scale = action_space.high[0]
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._act_noise_dumping = act_noise[0]
        self._act_noise_stddev = act_noise[1]
        self._ac_number = ac_number
        self._prior_weight = prior_weight

    def _make_actor(self):
        """Constructs and returns the actor model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        body = mlp(self._hidden_sizes, self._activation)(obs_input)
        mu = tf.keras.layers.Dense(self._act_dim)(body)

        # Normalize the actions.
        g = tf.reduce_mean(tf.math.abs(mu), axis=1, keepdims=True)
        g = tf.maximum(g, tf.ones_like(g))
        mu = mu / g

        # Add the action noise.
        pi = OUProcess(self._act_dim, self._act_noise_dumping,
                       self._act_noise_stddev)(mu)

        # Put the actions in the limit.
        mu = tf.tanh(mu) * self._act_scale
        pi = tf.tanh(pi) * self._act_scale

        return tf.keras.Model(inputs=obs_input, outputs=[mu, pi])

    def make_actor(self):
        """Constructs and returns the ensemble of actor models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        mus, pis = [], []
        for obs_input in tf.unstack(obs_inputs, axis=0):
            model = self._make_actor()
            mu, pi = model(obs_input)
            mus.append(mu)
            pis.append(pi)
        return tf.keras.Model(inputs=obs_inputs, outputs=[
            tf.stack(mus, axis=0),
            tf.stack(pis, axis=0),
        ])

    def _make_critic(self, trainable=True):
        """Constructs and returns the critic model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        act_input = tf.keras.Input(shape=(self._act_dim,))

        concat_input = tf.keras.layers.Concatenate(
            axis=-1)([obs_input, act_input])

        q = tf.keras.Sequential([
            mlp(self._hidden_sizes, self._activation, trainable=trainable),
            tf.keras.layers.Dense(1, trainable=trainable),
            tf.keras.layers.Reshape([])  # Very important to squeeze values!
        ])(concat_input)

        return tf.keras.Model(inputs=[obs_input, act_input], outputs=q)

    def make_critic(self):
        """Constructs and returns the ensemble of critic models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        act_inputs = tf.keras.Input(shape=(None, self._act_dim),
                                    batch_size=self._ac_number)
        qs = []
        for obs_input, act_input in zip(tf.unstack(obs_inputs, axis=0),
                                        tf.unstack(act_inputs, axis=0)):
            model = self._make_critic()
            q = model([obs_input, act_input])

            if self._prior_weight > 0:
                prior = self._make_critic(trainable=False)
                q += self._prior_weight * prior([obs_input, act_input])

            qs.append(q)
        return tf.keras.Model(inputs=[obs_inputs, act_inputs],
                              outputs=tf.stack(qs, axis=0))
