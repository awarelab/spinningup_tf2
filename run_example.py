"""Example of running SpinUp Bis algorithms."""
import argparse
import os

import gym
import tensorflow as tf

from spinup_bis import sac_tf2 as sac # pylint: disable=import-only-modules
from spinup_bis import sop_tf2 as sop # pylint: disable=import-only-modules


def env_fn():
    return gym.make('Pendulum-v0')


def _parse_args():
    parser = argparse.ArgumentParser(description='"Spinning Up TF2 example".')
    parser.add_argument('--output-dir', nargs='?', default="./out", help='Directory for saving checkpoints')
    parser.add_argument('--algorithm', nargs='?', default="sac", help='Currently supported: sac, sop')


    parser.parse_args()
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    save_path = args.output_dir + '/checkpoint'

    if args.algorithm == "sac":
        agent = sac
    elif args.algorithm == "sop":
        agent = sop
    else:
        raise ValueError("Algorithm not supported!")

    if 'NEPTUNE_PROJECT_NAME' in os.environ:
        neptune_kwargs = dict(project=os.environ['NEPTUNE_PROJECT_NAME'])
    else:
        neptune_kwargs = None

    ac_kwargs = dict(hidden_sizes=[16, 16],
                     activation=tf.nn.relu)

    logger_kwargs = dict(output_dir=args.output_dir,
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
          save_path=save_path)
