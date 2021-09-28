#!/usr/bin/env python
"""Tool for running rendered interactions of models with environments.

Requirements (pip):
  * gym
  * tensorflow
"""

import argparse

import gym
import tensorflow as tf


def run_model(model, env, episodes, print_reward):
    """Runs the given model on the provided Gym environment."""
    act_limit = env.action_space.high[0]

    for i_episode in range(episodes):
        observation = env.reset()
        done = False
        step_num = 0
        reward_sum = 0

        print(f'Starting episode #{i_episode}')

        while not done:
            env.render()

            if print_reward:
                print(f'timestep: {step_num}, reward = {reward_sum}')

            # WARNING: IT IS FOR SAC'S ACTOR, MODIFY IT WHEN REQUIRED!
            action = model(tf.expand_dims(observation, 0))[0][0]
            action = tf.clip_by_value(action, -act_limit, act_limit)

            observation, reward, done, _ = env.step(action)
            step_num += 1
            reward_sum += reward

        print(f'Episode finished after {step_num} timesteps')

    env.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Runs the saved model on the selected Gym environment.'
    )
    parser.add_argument(
        '--model_path', required=True, type=str,
        help='Path to the saved keras model of the agent.'
    )
    parser.add_argument(
        '--env_name', required=True, type=str,
        help='Name of the environment on which to run the model.'
    )
    parser.add_argument(
        '--num_episodes', default=10, type=int,
        help='Number of episodes to run the model for.'
    )
    parser.add_argument(
        '--skip_print', action='store_true',
        help='Flag specifying whether to skip printing the cumulative reward.'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    print_reward = not args.skip_print

    agent = tf.keras.models.load_model(args.model_path)
    env = gym.make(args.env_name)

    run_model(agent, env, args.num_episodes, print_reward)
