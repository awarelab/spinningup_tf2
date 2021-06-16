"""TD3 algorithm implementation."""

import random
import time

import numpy as np
import tensorflow as tf

from spinup_bis.algos.tf2.td3 import core
from spinup_bis.utils import logx


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=None, seed=0,
        total_steps=int(1e6), log_every=int(1e4), replay_size=int(1e6),
        gamma=0.99, polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100,
        start_steps=10000, update_after=1000, update_every=50,
        act_noise=0.1, max_ep_len=1000, num_test_episodes=10,
        logger_kwargs=None, save_freq=int(1e4), policy_update_delay=2,
        policy_noise_scale=0.2, policy_noise_clip=0.5, save_path=None):
    """
    Twin Delayed Deep Deterministic policy gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in `action_space` kwargs
            and returns actor and critic tf.keras.Model-s.

            Actor should take an observation in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ===========  ================  =====================================

            Critic should take an observation and action in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``q``        (batch,)          | Gives the current estimate of Q*
                                           | state and action in the input.
            ===========  ================  =====================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to TD3.

        seed (int): Seed for random number generators.

        total_steps (int): Number of environment interactions to run and train
            the agent.

        log_every (int): Number of environment interactions that should elapse
            between dumping logs.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of environment iterations) to save
            the current policy.

        save_path (str): The path specifying where to save the trained actor
            model (note: path needs to point to a directory). Setting the value
            to None turns off the saving.

        policy_update_delay (int): How often (in terms of regular updates)
            to update policy and target networks.

        policy_noise_scale (float): Standard deviation of policy smoothing
            noise.

        policy_noise_clip (float): Maximum absolute value of policy smoothing
            noise.

    """
    config = locals()
    logger = logx.EpochLogger(**(logger_kwargs or {}))
    logger.save_config(config)

    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs = ac_kwargs or {}
    ac_kwargs['action_space'] = env.action_space

    # Network
    actor, critic1, critic2 = actor_critic(**ac_kwargs)

    actor.build(input_shape=(None, obs_dim))
    critic1.build(input_shape=(None, obs_dim + act_dim))
    critic2.build(input_shape=(None, obs_dim + act_dim))

    # Target networks
    actor_targ, critic1_targ, critic2_targ = actor_critic(**ac_kwargs)

    actor_targ.build(input_shape=(None, obs_dim))
    critic1_targ.build(input_shape=(None, obs_dim + act_dim))
    critic2_targ.build(input_shape=(None, obs_dim + act_dim))

    # Copy weights
    actor_targ.set_weights(actor.get_weights())
    critic1_targ.set_weights(critic1.get_weights())
    critic2_targ.set_weights(critic2.get_weights())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim,
                                 act_dim=act_dim,
                                 size=replay_size)

    # Separate train ops for pi, q
    actor_opt = tf.keras.optimizers.Adam(learning_rate=pi_lr)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=q_lr)

    @tf.function
    def get_action(o, noise_scale):
        a = actor(tf.expand_dims(o, 0))[0]
        a += noise_scale * tf.random.normal(tf.shape(input=a))
        return tf.clip_by_value(a, -act_limit, act_limit)

    @tf.function
    # pylint: disable=unused-argument
    def delayed_update(obs1, obs2, acts, rews, done):
        with tf.GradientTape(persistent=True) as g:
            # Actor update
            pi = actor(obs1)
            q_pi = critic1(tf.concat([obs1, pi], -1))
            pi_loss = -tf.reduce_mean(q_pi)

        actor_grad = g.gradient(pi_loss, actor.trainable_variables)
        actor_opt.apply_gradients(
            zip(actor_grad, actor.trainable_variables)
        )

        for v, target_v in zip(actor.trainable_variables,
                               actor_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)
        for v, target_v in zip(critic1.trainable_variables,
                               critic1_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)
        for v, target_v in zip(critic2.trainable_variables,
                               critic2_targ.trainable_variables):
            target_v.assign(polyak * target_v + (1 - polyak) * v)

        return dict(
            pi_loss=pi_loss,
        )

    @tf.function
    def learn_on_batch(obs1, obs2, acts, rews, done):
        with tf.GradientTape(persistent=True) as g:
            # Critic update
            q1 = critic1(tf.concat([obs1, acts], -1))
            q2 = critic2(tf.concat([obs1, acts], -1))

            pi_targ = actor_targ(obs2)
            eps = policy_noise_scale * tf.random.normal(tf.shape(input=pi_targ))
            eps = tf.clip_by_value(eps, -policy_noise_clip, policy_noise_clip)
            pi_targ = tf.clip_by_value(pi_targ + eps, -act_limit, act_limit)

            q_pi_targ = tf.math.minimum(
                critic1_targ(tf.concat([obs2, pi_targ], -1)),
                critic2_targ(tf.concat([obs2, pi_targ], -1)),
            )

            # Bellman backup for Q function
            td_target = tf.stop_gradient(rews + gamma * (1 - done) * q_pi_targ)
            q1_loss = tf.reduce_mean((q1 - td_target) ** 2)
            q2_loss = tf.reduce_mean((q2 - td_target) ** 2)

        critic1_grad = g.gradient(q1_loss, critic1.trainable_variables)
        critic_opt.apply_gradients(
            zip(critic1_grad, critic1.trainable_variables)
        )

        critic2_grad = g.gradient(q2_loss, critic2.trainable_variables)
        critic_opt.apply_gradients(
            zip(critic2_grad, critic2.trainable_variables)
        )

        return dict(
            q1_loss=q1_loss,
            q2_loss=q2_loss,
            q1=q1,
            q2=q2,
        )

    def test_agent():
        for _ in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and (t + 1) % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)

                # Update the networks
                results = learn_on_batch(**batch)

                # Delayed update
                if (t + 1) % (update_every * policy_update_delay) == 0:
                    du_results = delayed_update(**batch)
                    logger.store(
                        LossPi=du_results['pi_loss'],
                    )

                logger.store(
                    LossQ1=results['q1_loss'],
                    LossQ2=results['q2_loss'],
                    Q1Vals=results['q1'],
                    Q2Vals=results['q2'],
                )

        # End of epoch wrap-up
        if ((t + 1) % log_every == 0) or (t + 1 == total_steps):
            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

        # Save model
        if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
            if save_path is not None:
                tf.keras.models.save_model(actor, save_path)
