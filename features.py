import tensorflow as tf
import tf_utils as tfu
import pdb


class BasePhi(tfu.Model):

    scope_name = 'Phi'

    def __init__(self, opts):
        super(BasePhi, self).__init__(opts)
        self.session = tfu.make_session()

        K, D = opts.dim_y, opts.dim_nu

        with tf.variable_scope(self.scope_name) as self.scope:
            globals().update(
                tfu.make_placeholders({
                    'x_pl': (None, opts.dim_x),
                    'C_pl': (K, K),
                    'y_true_pl': (None, K),
                    'y_down_pl': (None, K),
                    'y_up_pl': (None, K),
                })
            )

            train_feats = self._features(x_pl, is_training=True)
            test_feats = self._features(x_pl, is_training=False)
            phi_xy_train = self._phi(train_feats)
            phi_xy_test = self._phi(test_feats)

            C_aug_train = self._cost_matrix(C_pl, phi_xy_train, y_true_pl)
            C_aug_test = self._cost_matrix(
                C_pl, phi_xy_test, tf.zeros([tf.shape(x_pl)[0], K])
            )

            loss = tf.reduce_mean(
                tf.einsum('bi,bij,bj->b', y_up_pl, C_aug_train, y_down_pl)
            )
            it, alpha, train_op = self.make_train_op(loss)
            train_outputs = tfu.struct(
                it=it, lr=alpha, train_op=train_op, loss=loss,
            )

            self.functions = [
                ('phi', {'x': x_pl}, phi_xy_train),

                ('C',
                 {'c': C_pl, 'x': x_pl, 'y': y_true_pl},
                 C_aug_train),

                ('C_test',
                 {'c': C_pl, 'x': x_pl, 'y': y_true_pl},
                 C_aug_test),

                ('train',
                 {'c': C_pl, 'x': x_pl, 'yt': y_true_pl,
                  'yd': y_down_pl, 'yu': y_up_pl},
                 train_outputs),
            ]
            self.finalize()

    def _cost_matrix(self, C, phi, y_true):
        K = self.opts.dim_y
        ones = tf.ones([K])
        self.nu = nu = tfu.scoped_variable(
            'nu', '', shape=(self.opts.dim_nu),
            initializer=tf.truncated_normal_initializer(stddev=1e-2),
        )
        psi = tf.einsum('k,d,bdj->bkj', ones, nu,
                        phi - tf.einsum('bdk,bk,j->bdj', phi, y_true, ones))
        return C + psi

    def _phi(self, xx):
        K = self.opts.dim_y
        zeros = tf.zeros_like(xx)
        phi_xy = tf.stack([
            tf.concat([xx if i == j else zeros
                       for j in range(K)], axis=1)
            for i in range(K)
        ], axis=2)
        return phi_xy


class BasicPhi(BasePhi):

    def _features(self, x, is_training=True):
        assert x.get_shape().ndims == 2
        K = self.opts.dim_y
        dim_x = x.get_shape()[1].value
        xx_outer = tf.reshape(tf.einsum('bi,bj->bij', x, x),
                              [tf.shape(x)[0], dim_x * dim_x])
        return xx_outer


class DeepPhi(BasePhi):

    def _features(self, x, is_training=True):
        K = self.opts.dim_y
        dim_x = x.get_shape()[1].value

        x = tf.reshape(x, [tf.shape(x)[0], 32, 32, 3])
        layers = list(self.opts.sizes)
        fc_layers = layers.pop()

        for i, param in enumerate(layers):
            if is_training and param.get('dropout'):
                x = tf.nn.dropout(x, param['dropout'])

            x = tfu.conv(x, param, i, 'phi')
            if param.get('batch_norm'):
                x = tfu.batch_norm(
                    x, param['batch_norm'], is_training,
                    name='bn_%d' % i, scope_name='phi'
                )

            x = tfu.leaky_relu(x, leak=0.1)

            if param.get('pool'):
                x = tfu.pool(x, param['pool'])

            print x

        x = tfu.ravel(x)
        for i, param in enumerate(fc_layers):
            x = tfu.leaky_relu(
                tfu.affine(x, param['size'], 'fc_%d' % i, 'phi'),
                leak=0.1,
            )
            if is_training and param.get('dropout'):
                x = tf.nn.dropout(x, param['dropout'])

            print x
        return tfu.affine(x, self.opts.dim_phi, 'out', 'phi')


class DeepBaseline(DeepPhi):

    scope_name = 'baseline'

    def __init__(self, opts):
        super(BasePhi, self).__init__(opts)
        self.session = tfu.make_session()

        K, D = opts.dim_y, opts.dim_nu

        with tf.variable_scope(self.scope_name) as self.scope:
            globals().update(
                tfu.make_placeholders({
                    'x_pl': (None, opts.dim_x),
                    'y_true_pl': (None, K),
                })
            )

            train_feats = tf.nn.relu(
                self._features(x_pl, is_training=True)
            )
            test_feats = tf.nn.relu(
                self._features(x_pl, is_training=False)
            )
            y_hat_train = tfu.affine(train_feats, K, 'logits', 'phi')
            y_hat_test = tf.nn.softmax(
                tfu.affine(test_feats, K, 'logits', 'phi')
            )

            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=y_hat_train, labels=y_true_pl
                )
            )

            it, alpha, train_op = self.make_train_op(loss)
            train_outputs = tfu.struct(
                it=it, lr=alpha, train_op=train_op, loss=loss,
            )

            self.functions = [
                ('train', {'x': x_pl, 'y': y_true_pl},
                 train_outputs),

                ('test', {'x': x_pl}, y_hat_test),
            ]
            self.finalize()
