# coding=utf-8

from . import ops
from . import graph

__all__ = ['GradientDescentOptimizer', 'AdamOptimizer']


class OptimizeOp(ops.Operation):
    """这个 Operation 不执行普通意义上的计算，而是抽象了对模型的训练

    所有以梯度下降为核心操作的优化器都共用这个相同的 Operation"""
    def __init__(self, optimizer, loss, name='Optimizer'):
        super().__init__(name=name)
        self._optimizer = optimizer
        self._loss = loss

    def forward(self):
        """就是算一下梯度然后更新变量罢了=。="""
        gradients = self._optimizer.compute_gradients(self._loss)
        self._optimizer.apply_gradients(gradients)

    def grad(self, partial_op=None):
        raise NotImplementedError


class Optimizer(object):
    """优化器，用于降低代价函数

    这个类及其子类的实例都不是 Tensor 对象，不能用于 Session.run
    ——不过这个父类好像没啥用"""
    pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self._learning_rate = learning_rate
        self._graph = graph.get_default_graph()

    def minimize(self, loss) -> OptimizeOp:
        """返回一个 Tensor 对象，运行之即可对模型进行训练"""
        return OptimizeOp(self, loss)

    def compute_gradients(self, loss) -> dict:
        """计算并返回梯度"""
        gradients = {}
        for variable in self._graph.get_trainable_variables_collection():
            gradients[variable] = loss.grad(variable)
        return gradients

    def apply_gradients(self, gradients):
        """根据梯度对变量进行更新"""
        for variable in gradients:
            delta = gradients[variable] * self._learning_rate
            variable.set_value(variable.get_value() - delta)


class AdamOptimizer(GradientDescentOptimizer):
    # 假装已经写好了 =。=
    pass
