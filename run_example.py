"""Example of running SpinUp Bis algorithms."""

import gym
import tensorflow as tf

from spinup_bis import sac_tf2 as agent  # pylint: disable=import-only-modules


def env_fn():
    return gym.make('Pendulum-v0')


ac_kwargs = dict(hidden_sizes=[16, 16],
                 activation=tf.nn.relu)

logger_kwargs = dict(output_dir='out',
                     exp_name='Watch out, Pendulum!')

agent(env_fn=env_fn,
      ac_kwargs=ac_kwargs,
      total_steps=200_000,
      log_every=2000,
      replay_size=10_000,
      start_steps=1_000,
      update_after=1_000,
      max_ep_len=200,
      logger_kwargs=logger_kwargs)
