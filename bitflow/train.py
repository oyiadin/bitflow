# coding=utf-8

import numpy as np
from . import ops
from . import graph

__all__ = ['GradientDescentOptimizer', 'AdamOptimizer']


class OptimizeOp(ops.Operation):
    """这个 Operation 不执行普通意义上的计算，而是抽象了对模型的训练

    所有以梯度下降为核心操作的优化器都共用这个相同的 Operation"""
    def __init__(self, optimizer, loss, name='optimizer'):
        super().__init__(name=name)
        self._optimizer = optimizer
        self._loss = loss

    def forward(self):
        """就是算一下梯度然后更新变量罢了=。="""
        gradients = self._optimizer.compute_gradients(self._loss)
        self._optimizer.do_preparation(gradients)  # 可能会更新 gradients 的
        self._optimizer.apply_gradients(gradients)

    def grad(self, partial_op=None):
        raise NotImplementedError


class Optimizer(object):
    """优化器，用于降低代价函数

    这个类及其子类的实例都不是 Tensor 对象，不能用于 Session.run"""
    def __init__(self, name='Optimizer'):
        self.name = name
        self._graph = graph.get_default_graph()


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, name='GradientDescentOptimizer'):
        super().__init__(name=name)
        self._learning_rate = learning_rate

    def minimize(self, loss) -> OptimizeOp:
        """返回一个 Tensor 对象，运行之即可对模型进行训练"""
        return OptimizeOp(self, loss, name=self.name)

    def compute_gradients(self, loss) -> dict:
        """计算并返回梯度"""
        gradients = {}
        for variable in self._graph.get_trainable_variables_collection():
            gradients[variable] = loss.grad(variable)
        return gradients

    def do_preparation(self, gradients: dict):
        return

    def apply_gradients(self, gradients):
        """根据梯度对变量进行更新"""
        for variable in gradients:
            delta = gradients[variable] * self._learning_rate
            variable.set_value(variable.get_value() - delta)


class AdagradOptimizer(GradientDescentOptimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8, name='AdagradOptimizer'):
        super().__init__(learning_rate, name=name)
        self._epsilon = epsilon
        self._accumulators = {}

    def do_preparation(self, gradients: dict):
        for k, v in gradients.items():
            self._accumulators[k] = \
                self._accumulators.get(k, np.zeros_like(v)) + v ** 2

    def apply_gradients(self, gradients):
        for variable in gradients:
            real_rate = self._learning_rate / (
                    np.sqrt(self._accumulators[variable]) + self._epsilon)
            delta = gradients[variable] * real_rate
            variable.set_value(variable.get_value() - delta)


class MomentumOptimizer(GradientDescentOptimizer):
    """SGD 加上动量（惯性）的标准/nesterov版本"""
    def __init__(self, learning_rate=0.01, momentum=0.5, use_nesterov=False,
                 name='MomentumOptimizer'):
        super().__init__(learning_rate, name=name)
        self._momentum = momentum
        self._nesterov = use_nesterov
        self._velocity = {}

    def do_preparation(self, gradients: dict):
        """**就地**修改参数 gradients，并其返回修改后的值"""
        for k, v in gradients.items():
            delta = self._momentum * self._velocity.get(k, np.zeros_like(v))
            gradients[k] = self._velocity[k] = v + delta

    def compute_gradients(self, loss):
        if not self._nesterov:
            return super().compute_gradients(loss)

        # 先保存现场 XD
        trainable_variables = self._graph.get_trainable_variables_collection()
        _old_values = {i: i.get_value() for i in trainable_variables}

        # 挪到仅依靠惯性所能滚到的地方
        for i in trainable_variables:
            delta = self._learning_rate * self._momentum \
                    * self._velocity.get(i, np.zeros_like(_old_values[i]))
            i.set_value(_old_values[i] + delta)

        # 计算此处的梯度
        gradients = super().compute_gradients(loss)

        # 回去原来的位置，然后返回刚才的梯度
        for k, v in _old_values.items():
            k.set_value(v)
        return gradients


class AdamOptimizer(GradientDescentOptimizer):
    # 假装已经写好了 =。=
    pass


# alias
SGD = StochasticGradientDescentOptimizer = GradientDescentOptimizer
SGDM = MomentumOptimizer
NAG = lambda *args, **kwargs: \
    MomentumOptimizer(*args, **kwargs, use_nesterov=True)
Adam = AdamOptimizer
Adagrad = AdagradOptimizer
