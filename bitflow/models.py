# coding=utf-8

# 只有这个文件里的代码能正确实现矩阵求导（手工的）
# 当 y = f(x) 时，nn 里的函数仅能正确实现 ∂y/∂x
# 其中 y: scalar, x: scalar/vector/matrix
# 当然，nn 里的函数可以正确实现标量求导

import numpy as np
from . import nn
from . import ops
from . import train
from . import session

__all__ = ['LinearRegression']


class Model(ops.Tensor):
    """我是个魔鬼，把线性回归器之类封装好的模型也看作是 Tensor 对象 XD

    这里先写上一些有别于 Tensor 的额外接口"""
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def grad(self, partial_op=None):
        """矩阵求导仅会在这个层面（以及激活函数）进行，不会往深处探

        对于每个模型，我都会手动求导，并将求导结果事先写死在代码里"""
        raise NotImplementedError

    def predict(self, pred_x):
        raise NotImplementedError


class _DenseLayer(ops.Tensor):
    """由 R ^ units[0] --> R ^ units[1] 的一层 M-P 神经元模型"""
    def __init__(self, units: tuple, activation, name='dense_layer'):
        super().__init__(name=name)
        try:
            assert isinstance(units, (tuple, list))
            assert len(units) == 2
            assert isinstance(units[0], int)
            assert isinstance(units[1], int)
        except AssertionError:
            raise ValueError("`units` must be a tuple or a list containing "
                             "two integers")

        self._activation = activation
        self._X = ops.Placeholder()
        self._y = ops.Placeholder()

        self._W = ops.Variable(np.random.rand(*units))
        self._b = ops.Variable(np.random.rand(1, units[1]))
        # b 在参与计算时产生了 broadcast

        self._z = self._X @ self._W + self._b
        self._prediction = activation(self._z)

    def forward(self):
        return self._prediction.forward()

    def grad(self, partial_op=None):
        raise NotImplementedError


class LinearRegression(Model):
    def __init__(self, units: tuple, learning_rate=0.01,
                 name='linear_regression'):
        super().__init__(name=name)

        self._layer = _DenseLayer(units, nn.Raw)
        self._loss = nn.reduce_sum((self._layer - self._layer._y) ** 2)
        self._optimizer = train.GradientDescentOptimizer(
            learning_rate).minimize(self)

        self._fitted = False

    def fit(self, train_x, train_y, batch_size=5, epochs='auto',
            prompt_per_epochs=100, stop_when_delta=0.001):
        """对数据集进行拟合"""
        self._fitted = True
        # to ensure we are currently running within a Session
        sess = session.get_current_session()
        epoch = 0

        def do_train_for_one_epoch():
            nonlocal epoch
            epoch += 1
            if prompt_per_epochs and not epoch % prompt_per_epochs:
                print('Epoch #{:0>4}, loss={}'.format(
                    epoch, sess.run(self._loss, feed_dict={
                        self._layer._X: train_x,
                        self._layer._y: train_y})))
            for batch in range(0, train_x.shape[0], batch_size):
                sess.run(self._optimizer, feed_dict={
                    self._layer._X: train_x[batch:batch + batch_size],
                    self._layer._y: train_y[batch:batch + batch_size]})

        if epochs == 'auto':
            # 落到极小值时自动结束
            do_train_for_one_epoch()

            feed_dict = {
                self._layer._X: train_x, self._layer._y: train_y}
            last_loss = sess.run(self._loss, feed_dict=feed_dict)
            while True:
                do_train_for_one_epoch()
                now_loss = sess.run(self._loss, feed_dict=feed_dict)
                delta = (last_loss - now_loss) / last_loss
                last_loss = now_loss
                if delta < 0:
                    if epoch <= 2:
                        raise RuntimeWarning(
                            "loss function appears to be rising up, stopping\n"
                            "  [note] or you can apply a smaller learning rate")
                    else:
                        return
                if delta < stop_when_delta:
                    return
                    # 当损失函数的改变率低于 0.1% 后就自动结束

        else:
            assert isinstance(epochs, int)
            for epoch in range(epochs):
                do_train_for_one_epoch()

    def grad(self, partial_op=None):
        # 我没法实现自动进行矩阵求导，只能以一整块公式作为整体
        # 事先手工计算，并在此写死在代码里边
        # grad 的调用不会传播到 matmul 之类最底层的操作
        X = self._layer._X.forward()
        W = self._layer._W.forward()
        b = self._layer._b.forward()
        tile_shape = list(b.shape)
        tile_shape[0] = X.shape[0]
        b = np.tile(b, tile_shape)
        y = self._layer._y.forward()
        if partial_op == self._layer._W:
            return 2 * X.T @ X @ W + 2 * X.T @ (b - y)
        elif partial_op == self._layer._b:
            _temp = X @ W + 2 * (b - y)
            return sum((i for i in _temp))
        else:
            return np.zeros_like(partial_op.forward())

    def predict(self, pred_x):
        if not self._fitted:
            raise RuntimeError('you must call fit() before do any prediction')

        sess = session.get_current_session()
        return sess.run(
            self._layer, feed_dict={self._layer._X: pred_x})
