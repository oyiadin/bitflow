#coding=utf-8

from . import ops
from . import graph

__all__ = ['GradientDescentOptimizer']


class OptimizeOp(ops.Operation):
    def __init__(self, optimizer, loss, name='Optimizer'):
        self._optimizer = optimizer
        self._loss = loss
        super().__init__(name=name)

    def forward(self):
        gradients = self._optimizer.compute_gradients(self._loss)
        self._optimizer.apply_gradients(gradients)

    def grad(self, partial_derivative_op=None):
        raise NotImplementedError


class Optimizer(object):
    '''优化器，用于降低代价函数。'''
    pass


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__()
        self._learning_rate = learning_rate
        self._graph = graph.get_default_graph()

    def minimize(self, loss):
        return OptimizeOp(self, loss)

    def compute_gradients(self, loss):
        gradients = {}
        for variable in self._graph.get_trainable_variables_collection():
            gradients[variable] = loss.grad(variable)
        return gradients

    def apply_gradients(self, gradients):
        for variable in gradients:
            delta = gradients[variable] * self._learning_rate
            variable.set_value(variable.get_value() - delta)
