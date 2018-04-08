import tensorflow as tf


class DenseQNet(object):
    """
    Deep Q Network with only fully connected hidden layers.
    The main network is used for the training,
    while the others just provide inference.
    """

    def __init__(self,
                 n_actions,
                 n_hidden_layers,
                 w_hidden_layers,
                 activation,
                 state_shape,
                 is_main=False,
                 name='DeepQNet'):

        super(DenseQNet, self).__init__()
        self.n_actions = n_actions
        self.n_hidden_layers = n_hidden_layers
        self.w_hidden_layers = w_hidden_layers
        self.activation = activation
        self.state_shape = state_shape
        self.is_main = is_main
        self.name = name

        self.n_childs = 0

        with tf.variable_scope(name):
            with tf.variable_scope('placeholders'):
                self.inputs = tf.placeholder(dtype=tf.float32,
                                             shape=[None] + state_shape,
                                             name='input_state')

                if is_main:
                    self.q_targets = tf.placeholder(dtype=tf.float32,
                                                    shape=[None],
                                                    name='target_Q_values')

                    self.actions = tf.placeholder(dtype=tf.int32,
                                                  shape=[None],
                                                  name='actions_taken')

            with tf.variable_scope('inputs'):
                self.inputs_flat = tf.layers.flatten(inputs=self.inputs,
                                                     name='flattened_inputs')

            previous_layer = self.inputs_flat
            with tf.variable_scope('hidden_layers'):

                for i in range(n_hidden_layers):
                    previous_layer = \
                        tf.layers.dense(inputs=previous_layer,
                                        units=w_hidden_layers,
                                        activation=activation,
                                        kernel_initializer=tf.truncated_normal_initializer(0, 0.2),
                                        trainable=is_main,
                                        name='dense_%d' % i)

            # TODO : value and advantage separated (https://arxiv.org/pdf/1511.06581.pdf)
            with tf.variable_scope('outputs'):
                self.q_values = tf.layers.dense(inputs=previous_layer,
                                                units=n_actions,
                                                activation=None,
                                                kernel_initializer=tf.truncated_normal_initializer(0, 0.2),
                                                trainable=is_main,
                                                name='Q_values')

                self.pred_actions = tf.argmax(input=self.q_values,
                                              axis=1,
                                              name='actions')

            if is_main:
                with tf.variable_scope('training'):
                    self.actions_one_hot = tf.one_hot(indices=self.actions,
                                                      depth=n_actions)

                    self.chosen_q_values = tf.reduce_sum(self.q_values * self.actions_one_hot,
                                                         axis=1)

                    self.loss = tf.losses.mean_squared_error(predictions=self.chosen_q_values,
                                                             labels=self.q_targets)

    def update_weights(self, target_net):
        target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, target_net.name)
        network_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

        return [mine.assign(target) for target, mine in zip(target_weights, network_weights)]

    def make_child(self):
        self.n_childs += 1
        return DenseQNet(n_actions=self.n_actions,
                         n_hidden_layers=self.n_hidden_layers,
                         w_hidden_layers=self.w_hidden_layers,
                         activation=self.activation,
                         state_shape=self.state_shape,
                         is_main=False,
                         name=self.name + '_Child%d' % self.n_childs)


if __name__ == '__main__':
    main_net = DenseQNet(n_actions=4,
                         n_hidden_layers=10,
                         w_hidden_layers=60,
                         activation=tf.nn.relu,
                         state_shape=[4, 4],
                         is_main=True,
                         name='DenseQNet')

    child_net = main_net.make_child()

    with tf.Session() as sess:
        tf.summary.FileWriter(logdir='logdir',
                              graph=sess.graph)
