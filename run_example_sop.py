"""Example of running SpinUp Bis algorithms."""

import os
import gym
import tensorflow as tf

from spinup_bis import sop_tf2 as agent  # pylint: disable=import-only-modules


def env_fn():
    return gym.make('Pendulum-v0')


if 'NEPTUNE_PROJECT_NAME' in os.environ:
    neptune_kwargs = dict(project=os.environ['NEPTUNE_PROJECT_NAME'])
else:
    neptune_kwargs = None


ac_kwargs = dict(hidden_sizes=[16, 16],
                 activation=tf.nn.relu,
                 prior_weight=0.0)

logger_kwargs = dict(output_dir='out',
                     exp_name='Watch out, Pendulum!',
                     neptune_kwargs=neptune_kwargs)

agent(env_fn=env_fn,
      ac_kwargs=ac_kwargs,
      total_steps=200_000,
      log_every=2000,
      replay_size=10_000,
      start_steps=1_000,
      update_after=1_000,
      max_ep_len=200,
      logger_kwargs=logger_kwargs,
      save_freq=200_000,
      save_path='./out/checkpoint')
