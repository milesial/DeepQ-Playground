import random

import gym
import numpy as np
import tensorflow as tf

from dense_q.denseqnet import DenseQNet
from epsilon_decay import EpsilonDecay
from experience_buffer import ExperienceBuffer


class DenseGym(object):
    """
    Class for experimenting with the DenseQNet on openai gym environments
    """

    def __init__(self, env_id):
        super(DenseGym, self).__init__()

        self.env = gym.make(env_id)
        tf.logging.info('Now using gym env %s' % env_id)

        self.main_net = DenseQNet(n_actions=self.env.action_space.n,
                                  n_hidden_layers=2,
                                  w_hidden_layers=30,
                                  activation=tf.nn.relu,
                                  state_shape=self.env.observation_space.shape,
                                  is_main=True,
                                  name='MainDenseQNet')

        self.target_net = self.main_net.make_child(name='TargetDenseQNet')

        self.update_net_op = self.target_net.update_weights(target_net=self.main_net)

    def run(self,
            gamma,
            n_runs,
            max_steps,
            batch_size,
            buffer_size,
            learning_rate,
            update_interval,
            epsilon_duration):

        tf.logging.info(' Starting the loop...')
        tf.logging.info('   n_runs:    %d' % n_runs)
        tf.logging.info('   max_steps: %d' % max_steps)
        tf.logging.info('   e-decay:   %d' % epsilon_duration)
        tf.logging.info('   l. rate:   %f' % learning_rate)
        tf.logging.info('   batches:   %d' % batch_size)
        tf.logging.info('   gamma:     %f' % gamma)
        tf.logging.info('   buffer:    %d' % buffer_size)
        tf.logging.info('   update:    %d' % update_interval)

        epsilon = EpsilonDecay(start_value=1.0,
                               end_value=0.1,
                               duration=epsilon_duration)

        buffer = ExperienceBuffer(max_size=buffer_size)

        train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.main_net.loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for run in range(n_runs):
                cumulative_reward = 0
                cumulative_loss = 0

                state = self.env.reset()

                for step in range(max_steps):

                    # Run and populate buffer with a sample
                    e = epsilon.get()
                    if random.random() < e:
                        action = random.randrange(self.env.action_space.n)
                    else:
                        feed_dict = {self.main_net.inputs: [state]}
                        action = sess.run(self.main_net.pred_actions, feed_dict)[0]

                    state_next, reward, end, _ = self.env.step(action)
                    cumulative_reward += reward

                    buffer.append([state, action, reward, end, state_next])

                    state = state_next

                    if len(buffer) > batch_size:
                        samples = buffer.sample(batch_size)

                        targets = samples[:, 2]

                        # Double DQN (https://arxiv.org/pdf/1509.06461.pdf)
                        # To compute the target Q values
                        feed_dict = {self.main_net.inputs: np.vstack(samples[:, 4])}
                        actions_state_next = sess.run(self.main_net.pred_actions, feed_dict)

                        feed_dict = {self.target_net.inputs: np.vstack(samples[:, 4])}
                        target_q_values = sess.run(self.target_net.q_values, feed_dict)
                        targets[:] += \
                            (1 - samples[:, 3]) * target_q_values[
                                range(target_q_values.shape[0]), actions_state_next] * gamma

                        # Train the main network
                        feed_dict = {self.main_net.inputs: np.vstack(samples[:, 0]),
                                     self.main_net.actions: samples[:, 1],
                                     self.main_net.q_targets: targets}

                        loss, _ = sess.run([self.main_net.loss, train_op], feed_dict)
                        cumulative_loss += loss

                    if step % update_interval == 0:
                        sess.run(self.update_net_op)

                    if end:
                        break

                if run % int(n_runs / 100) == 0:
                    tf.logging.debug(' Run %d / %d, reward: %f, loss: %f, epsilon %f' % (run + 1, n_runs,
                                                                                         cumulative_reward,
                                                                                         cumulative_loss,
                                                                                         e))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)

    DenseGym(env_id='CartPole-v1') \
        .run(n_runs=10000,
             max_steps=1000,
             learning_rate=0.0003,
             epsilon_duration=20000,
             buffer_size=100000,
             batch_size=50,
             update_interval=20,
             gamma=0.95)
