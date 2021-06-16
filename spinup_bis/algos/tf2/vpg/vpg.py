"""VPG algorithm implementation."""

import random
import time

import numpy as np
import tensorflow as tf

from spinup_bis.algos.tf2.vpg import core
from spinup_bis.utils import logx
from spinup_bis.utils import mpi_tf
from spinup_bis.utils import mpi_tools


class VPGBuffer:
    """A buffer for storing trajectories experienced by a VPG agent.

    Uses Generalized Advantage Estimation (GAE-Lambda) for calculating
    the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim),
                                dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim),
                                dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Computes reward for unfinished trajectory.

        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = \
            core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """Returns data stored in buffer.

        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_tools.mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf,
                self.ret_buf, self.logp_buf]


def vpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=None, seed=0,
        total_steps=int(1e6), train_every=4000, log_every=4000, gamma=0.99,
        pi_lr=3e-4, v_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=None, save_freq=int(1e4), save_path=None):
    """Vanilla Policy Gradient with GAE-Lambda for advantage estimation.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in `action_space` and
            `observation_space` kwargs, and returns actor and critic
            tf.keras.Model-s.

            Actor implements method `action` which should take an observation
            in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ===========  ================  =====================================

            Furthermore, actor implements method `action_logprob` which should
            take an observation and an action in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``logp``     (batch,)          | Gives the log probability of
                                           | selecting each specified action.
            ===========  ================  =====================================

            Critic should take an observation in and output:
            ===========  ================  =====================================
            Symbol       Shape             Description
            ===========  ================  =====================================
            ``v``        (batch,)          | Gives estimate of current policy
                                           | value for states and actions in the
                                           | input.
            ===========  ================  =====================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to VPG.

        seed (int): Seed for random number generators.

        total_steps (int): Number of environment interactions to run and train
            the agent.

        train_every (int): Number of environment interactions that should elapse
            between training epochs.

        log_every (int): Number of environment interactions that should elapse
            between dumping logs.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        v_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        save_path (str): The path specifying where to save the trained actor
            model (note: path needs to point to a directory). Setting the value
            to None turns off the saving.
    """
    config = locals()
    logger = logx.EpochLogger(**(logger_kwargs or {}))
    logger.save_config(config)

    seed += 10000 * mpi_tools.proc_id()
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    mpi_num_proc = mpi_tools.num_procs()

    # In case of distributed computations these values have to be updated
    total_steps = core.distribute_value(total_steps, mpi_num_proc)
    train_every = core.distribute_value(train_every, mpi_num_proc)
    log_every = core.distribute_value(log_every, mpi_num_proc)
    save_freq = core.distribute_value(save_freq, mpi_num_proc)

    env = env_fn()
    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.shape
    env.seed(seed)

    replay_buffer = VPGBuffer(obs_dim=obs_dim, act_dim=act_dim,
                              size=train_every, gamma=gamma, lam=lam)

    ac_kwargs = ac_kwargs or {}
    ac_kwargs['observation_space'] = env.observation_space
    ac_kwargs['action_space'] = env.action_space

    actor, critic = actor_critic(**ac_kwargs)

    actor.build(input_shape=(None, obs_dim))
    critic.build(input_shape=(None, obs_dim))

    actor_optimizer = mpi_tf.MpiAdamOptimizer(learning_rate=pi_lr)
    critic_optimizer = mpi_tf.MpiAdamOptimizer(learning_rate=v_lr)

    mpi_tf.sync_params(actor.variables)
    mpi_tf.sync_params(critic.variables)

    @tf.function
    def value(observations):
        return critic(observations)

    def get_value(observation):
        return value(np.array([observation])).numpy()[0]

    @tf.function
    def value_loss(observations, rtg):
        return tf.reduce_mean((critic(observations) - rtg) ** 2)

    @tf.function
    def value_train_step(observations, rtg):
        def loss():
            return value_loss(observations, rtg)

        critic_optimizer.minimize(loss, critic.trainable_variables)
        mpi_tf.sync_params(critic.variables)

        return loss()

    def get_action(observation):
        return actor.action(np.array([observation])).numpy()[0]

    @tf.function
    def pi_loss(logp, advantages):
        return -tf.reduce_mean(tf.math.multiply(logp, advantages))

    @tf.function
    def pi_train_step(observations, actions, advantages, logp_old):
        def loss():
            logp = actor.action_logprob(observations, actions)
            return pi_loss(logp, advantages)

        actor_optimizer.minimize(loss, actor.trainable_variables)
        mpi_tf.sync_params(actor.variables)

        # For logging purposes
        logp = actor.action_logprob(observations, actions)
        loss_new = pi_loss(logp, advantages)

        return loss_new, tf.reduce_mean(logp_old - logp), tf.reduce_mean(-logp)

    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        action = get_action(obs)
        v_t = get_value(obs)
        logp = actor.action_logprob(np.array([obs]),
                                    np.array([action])).numpy()[0]

        # Step the env
        new_obs, rew, done, _ = env.step(action)
        ep_ret += rew
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        done = False if ep_len == max_ep_len else done

        # Store experience to replay buffer
        replay_buffer.store(obs, action, rew, v_t, logp)
        logger.store(VVals=v_t)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        obs = new_obs

        # End of trajectory handling
        if done or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)

        if done or (ep_len == max_ep_len) or (t + 1) % train_every == 0:
            obs, ep_ret, ep_len = env.reset(), 0, 0

            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = 0 if done else get_value(obs)
            replay_buffer.finish_path(last_val)

        # Update handling
        if (t + 1) % train_every == 0:
            [batch_obs, batch_act, batch_adv, batch_rtg,
             batch_logp] = replay_buffer.get()

            loss, kl, entropy = pi_train_step(batch_obs, batch_act,
                                              batch_adv, batch_logp)
            logger.store(LossPi=loss.numpy(), KL=kl.numpy(),
                         Entropy=entropy.numpy())

            for _ in range(train_v_iters):
                loss = value_train_step(batch_obs, batch_rtg)
                logger.store(LossV=loss)

        # End of epoch wrap-up
        if ((t + 1) % log_every == 0) or (t + 1 == total_steps):
            # Log info about epoch
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (t + 1) * mpi_num_proc)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

        # Save model
        if ((t + 1) % save_freq == 0) or (t + 1 == total_steps):
            if mpi_tools.proc_id() == 0 and save_path is not None:
                tf.keras.models.save_model(actor, save_path)
