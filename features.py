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
                'subgrad_pl': (D, K),
            }
            for cmd in tfu.make_placeholders(pl):
                exec(cmd)

            phi_xy = self._phi(x_pl)
            loss = self._loss(x_pl, subgrad_pl)

            if loss is None:
                train_outputs = tf.no_op()
            else:
                it, alpha, train_op = self.make_train_op(loss)
                train_outputs = tfu.struct(
                    it=it, alpha=alpha, train_op=train_op
                )

            self.functions = [
                ('phi', {'x': x_pl}, phi_xy),
                ('train', {'x': x_pl, 'g': g_pl}, train_outputs),
            ]
            self.finalize()

    def _grad(self, phi_xy, grad_pl):
        raise NotImplemented

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

    def _loss(self, x, g):
        return None


class DeepPhi(BasePhi):

    def _phi(self, x):
        for i, d in enumerate(self.opts.sizes):
            x = tf.tanh(
                tfu.affine(x, d, i, 'phi')
            )
        xx = tfu.affine(x, self.opts.dim_phi, 'out', 'phi')
        zeros = tf.zeros_like(xx)

        phi_xy = tf.stack([
            tf.concat([xx_outer if i == j else zeros
                       for j in range(K)], axis=1)
            for i in range(K)
        ], axis=2)

        return phi_xy

    def _loss(self, x, g):
        phi = self._phi(x)
        dummy_loss = tf.reduce_sum(phi * g)
        return dummy_loss
