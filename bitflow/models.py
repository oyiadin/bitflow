# coding=utf-8

import numpy as np
from . import nn
from . import ops
from . import train
from . import session

__all__ = ['LinearRegression']


class Model(ops.Tensor):
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, partial_derivative_op=None):
        raise NotImplementedError

    def predict(self, pred_x):
        raise NotImplementedError


class _DenseLayer(ops.Operation):
    def __init__(self, units, activation, name='dense_layer'):
        try:
            assert isinstance(units, (tuple, list))
            assert len(units) == 2
            assert isinstance(units[0], int)
            assert isinstance(units[1], int)
        except AssertionError:
            raise ValueError("`units` must be a tuple or a list containing "
                             "two integers")

        self._activation = activation
        self._X = ops.placeholder()
        self._y = ops.placeholder()

        self._W = ops.Variable(np.zeros(units))#np.random.rand(units[0], units[1]))
        self._b = ops.Variable(np.zeros(units))#np.random.rand(1, units[1]))

        self._z = self._X @ self._W + self._b
        self._prediction = activation(self._z)
        super().__init__(name=name)

    def forward(self):
        return self._prediction.forward()


class LinearRegression(Model):
    def __init__(self, units, learning_rate=0.01, name='linear_regression'):
        self._dense_layer = _dense_layer = _DenseLayer(units, nn.Raw)
        super().__init__(name=name)

        self._loss = nn.reduce_sum((_dense_layer - _dense_layer._y) ** 2)
        self._optimizer = train.GradientDescentOptimizer(
            learning_rate).minimize(self)
        self._fitted = False

    def fit(self, train_x, train_y, batch_size=5, epochs='auto'):
        self._fitted = True
        # to ensure we are currently running within a Session
        sess = session.get_current_session()

        def do_train_for_one_epoch():
            for batch in range(0, train_x.shape[0], batch_size):
                sess.run(self._optimizer, feed_dict={
                    self._dense_layer._X: train_x[batch:batch+batch_size],
                    self._dense_layer._y: train_y[batch:batch+batch_size]})

        if epochs == 'auto':
            do_train_for_one_epoch()

            feed_dict = {
                self._dense_layer._X: train_x, self._dense_layer._y: train_y}
            last_loss = sess.run(self._loss, feed_dict=feed_dict)
            while True:
                do_train_for_one_epoch()
                now_loss = sess.run(self._loss, feed_dict=feed_dict)
                delta = (last_loss - now_loss) / last_loss
                if delta < 0.1:
                    return

        else:
            assert isinstance(epochs, int)
            for epoch in range(epochs):
                do_train_for_one_epoch()

    def grad(self, partial_derivative_op=None):
        # 我没法实现自动进行矩阵求导，只能以一整块公式作为整体
        # 事先手工计算，并在此写死在代码里边
        # grad 的调用不会传播到 matmul 之类最底层的操作
        X = self._dense_layer._X.forward()
        W = self._dense_layer._W.forward()
        b = self._dense_layer._b.forward()
        tile_shape = list(b.shape)
        tile_shape[0] = X.shape[0]
        b = np.tile(b, tile_shape)
        y = self._dense_layer._y.forward()
        if partial_derivative_op == self._dense_layer._W:
            return 2 * X.T @ X @ W + 2 * X.T @ (b - y)
        elif partial_derivative_op == self._dense_layer._b:
            _temp = X @ W + 2 * (b - y)
            return sum((i for i in _temp))
        else:
            return np.zeros_like(partial_derivative_op.forward())

    def predict(self, pred_x):
        if not self._fitted:
            raise RuntimeError('you must call fit() before do any prediction')

        sess = session.get_current_session()
        return sess.run(
            self._dense_layer, feed_dict={self._dense_layer._X: pred_x})
