import tensorflow as tf
import tf_utils as tfu
import pdb


class BasePhi(tfu.Model):

    scope_name = 'Phi'

    def __init__(self, opts):
        super(BasePhi, self).__init__(opts)
        self.session = tfu.make_session(0.1)

        K, D = opts.dim_y, opts.dim_nu

        with tf.variable_scope(self.scope_name) as self.scope:
            pl = {
                'x_pl': (None, opts.dim_x),
                'C_pl': (K, K),
                'y_true_pl': (None, K),
                'y_down_pl': (None, K),
                'y_up_pl': (None, K),
            }
            for cmd in tfu.make_placeholders(pl):
                exec(cmd)

            phi_xy = self._phi(x_pl)
            C_aug = self._cost_matrix(C_pl, phi_xy, y_true_pl)

            loss = tf.reduce_mean(
                tf.einsum('bi,bij,bj->b', y_up_pl, C_aug, y_down_pl)
            )

            it, alpha, train_op = self.make_train_op(loss)
            train_outputs = tfu.struct(
                it=it, lr=alpha, train_op=train_op, loss=loss,
            )

            self.functions = [
                ('reset', {}, tf.global_variables_initializer()),

                ('phi', {'x': x_pl}, phi_xy),

                ('C',
                 {'c': C_pl, 'x': x_pl, 'y': y_true_pl},
                 C_aug),

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
            initializer=tf.zeros_initializer(tf.float32)
        )
        psi = tf.einsum('k,d,bdj->bkj', ones, nu,
                        phi - tf.einsum('bdk,bk,j->bdj', phi, y_true, ones))
        return C + psi

    def _phi(self, x):
        raise NotImplemented


class BasicPhi(BasePhi):

    def _phi(self, x):
        assert x.get_shape().ndims == 2
        K = self.opts.dim_y
        dim_x = x.get_shape()[1].value
        xx_outer = tf.reshape(tf.einsum('bi,bj->bij', x, x),
                              [tf.shape(x)[0], dim_x * dim_x])
        zeros = tf.zeros_like(xx_outer)

        phi_xy = tf.stack([
            tf.concat([xx_outer if i == j else zeros
                       for j in range(K)], axis=1)
            for i in range(K)
        ], axis=2)

        return phi_xy


class DeepPhi(BasePhi):

    def _phi(self, x):
        K = self.opts.dim_y
        for i, d in enumerate(self.opts.sizes):
            x = tfu.leaky_relu(
                tfu.affine(x, d, i, 'phi'),
                leak=0.1,
            )
        xx = tfu.affine(x, self.opts.dim_phi, 'out', 'phi')
        zeros = tf.zeros_like(xx)

        phi_xy = tf.stack([
            tf.concat([xx if i == j else zeros
                       for j in range(K)], axis=1)
            for i in range(K)
        ], axis=2)

        return phi_xy
