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
    """我是个魔鬼，把线性回归器之类封装好的模型也看作是 Tensor 对象 XD"""
    def __init__(self, name='Model'):
        super().__init__(name=name)
        self._fitted = False
        self._layer = None
        self._loss = None
        self._optimizer = None
        self.prediction = None

    def grad(self, partial_op=None):
        """矩阵求导仅会在这个层面（以及激活函数）进行，不会往深处探

        对于每个模型，我都会手动求导，并将求导结果事先写死在代码里"""
        raise NotImplementedError

    def predict(self, pred_x):
        if not self._fitted:
            raise RuntimeError('you must call fit() before do any prediction')

        sess = session.get_current_session()
        return sess.run(
            self.prediction, feed_dict={self.X: pred_x})

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
            for batch in range(0, train_x.shape[0], batch_size):
                sess.run(self._optimizer, feed_dict={
                    self.X: train_x[batch:batch + batch_size],
                    self.y: train_y[batch:batch + batch_size]})
            if prompt_per_epochs and not epoch % prompt_per_epochs:
                print('Epoch #{:0>4}, loss={}'.format(
                    epoch, sess.run(self._loss, feed_dict={
                        self.X: train_x,
                        self.y: train_y})))

        if epochs == 'auto':
            # 落到极小值时自动结束
            do_train_for_one_epoch()

            feed_dict = {
                self.X: train_x, self.y: train_y}
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


class _DenseLayer(ops.Tensor):
    """由 R ^ units[0] --> R ^ units[1] 的一层 M-P 神经元模型"""
    def __init__(self, units: tuple, activation, X=None, y=None,
                 name='_DenseLayer'):
        super().__init__(name=name)
        try:
            assert isinstance(units, (tuple, list))
            assert len(units) == 2
            assert isinstance(units[0], int)
            assert isinstance(units[1], int)
        except AssertionError:
            raise ValueError("`units` must be a tuple or a list containing "
                             "two integers")

        self.activation = activation
        self.X = X or ops.Placeholder()
        self.y = y or ops.Placeholder()

        self.W = ops.Variable(np.random.rand(*units))
        self.b = ops.Variable(np.random.rand(1, units[1]))
        # b 在参与计算时产生了 broadcast

        self.z = self.X @ self.W + self.b
        self.prediction = activation(self.z)

    def forward(self):
        return self.prediction.forward()

    def back_prop(self, partial_op=None, delta=None):
        if delta is None:  # 在最后一层
            delta = self.forward() - self.y.forward()

        if partial_op == self.W:
            return self.X.T.forward() @ delta
        elif partial_op == self.b:
            return np.ones((1, delta.shape[0])) @ delta

        else:
            if isinstance(self.X, _DenseLayer):  # 如果前边还有其他层
                new_delta = (delta @ self.W.T.forward()) \
                            * self.X.forward() * (1 - self.X.forward())
                return self.X.back_prop(partial_op, delta=new_delta)
            else:
                return np.zeros_like(partial_op.forward())


class LinearRegression(Model):
    def __init__(self, units: tuple, optimizer=train.GradientDescentOptimizer,
                 name='LinearRegression', **kwargs):
        super().__init__(name=name)

        self._layer = _DenseLayer(units, nn.Raw)
        self.X = self._layer.X
        self.y = self._layer.y
        self._loss = nn.reduce_sum((self._layer - self._layer.y) ** 2)
        self._optimizer = optimizer(**kwargs).minimize(self)
        self.prediction = self._layer.prediction

    def grad(self, partial_op=None):
        # 我没法实现自动进行矩阵求导，只能以一整块公式作为整体
        # 事先手工计算，并在此写死在代码里边
        # grad 的调用不会传播到 matmul 之类最底层的操作
        X = self._layer.X.forward()
        W = self._layer.W.forward()
        b = self._layer.b.forward()
        tile_shape = list(b.shape)
        tile_shape[0] = X.shape[0]
        b = np.tile(b, tile_shape)
        y = self._layer.y.forward()
        if partial_op == self._layer.W:
            return 2 * X.T @ X @ W + 2 * X.T @ (b - y)
        elif partial_op == self._layer.b:
            _temp = X @ W + 2 * (b - y)
            return sum((i for i in _temp))
        else:
            return np.zeros_like(partial_op.forward())


class LogisticRegression(Model):
    """仅支持二分类"""
    def __init__(self, units: tuple, optimizer=train.GradientDescentOptimizer,
                 name='LogisticRegression', **kwargs):
        super().__init__(name=name)
        assert len(units) == 1 or units[1] == 1
        # 仅支持二分类，所以确保要么省略输出维度，要么 units[1] == 1

        self._layer = _DenseLayer(units, nn.sigmoid)
        self.X = self._layer.X
        self.y = self._layer.y
        self._loss = - (self.y.T @ ops.LogOp(self._layer)) \
                     - (1 - self.y).T @ ops.LogOp(1 - self._layer)
        self._optimizer = optimizer(**kwargs).minimize(self)
        self.prediction = self._layer.prediction

    def grad(self, partial_op=None):
        X = self._layer.X.forward()
        y = self._layer.y.forward()
        pred = self._layer.forward()
        if partial_op == self._layer.W:
            return X.T @ (pred - y)
        elif partial_op == self._layer.b:
            _temp = pred - y
            return sum((i for i in _temp))
        else:
            return np.zeros_like(partial_op.forward())


class DenseLayers(Model):
    def __init__(self, units: tuple, optimizer=train.GradientDescentOptimizer,
                 name='DenseLayers', **kwargs):
        super().__init__(name=name)
        assert isinstance(units, (tuple, list)) and isinstance(units[0], int)
        assert len(units) >= 2

        self.layers = []
        X = None
        y = ops.Placeholder()

        for n in range(0, len(units)-1):
            layer = _DenseLayer(units=units[n:n+2], activation=nn.sigmoid,
                                X=X, y=y)
            self.layers.append(layer)
            X = layer

        self.X = self.layers[0].X
        self.y = y
        self._loss = - (self.y.T @ ops.LogOp(self.layers[-1])) \
                     - (1 - self.y).T @ ops.LogOp(1 - self.layers[-1])
        self._optimizer = optimizer(**kwargs).minimize(self)
        self.prediction = self.layers[-1]

    def grad(self, partial_op=None):
        return self.layers[-1].back_prop(partial_op)
