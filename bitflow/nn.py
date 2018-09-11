# coding=utf-8

# 其他代码对 nn 所有函数做的保证还没实现呢 XD
# 等我下一个 commit 吧～

import numpy as np
from . import ops

__all__ = ['reduce_sum', 'sigmoid']


class Raw(ops.Operation):
    def __init__(self, input, name='raw'):
        super().__init__(name=name)
        self._input = input

    def forward(self):
        return self._input.forward()

    def grad(self, partial_op=None):
        return np.ones_like(partial_op.forward())


class _ReduceSum(ops.Operation):
    def __init__(self, input, name='reduce_sum'):
        super().__init__(name=name)
        self._input = input

    def forward(self):
        return sum((i for i in self._input.forward()))

    def grad(self, partial_op=None):
        _temp = self._input.grad(partial_op)
        return _temp


class _Sigmoid(ops.Operation):
    def __init__(self, input, name='sigmoid'):
        super().__init__(name=name)
        self._input = input
        c1 = ops.Constant(value=1)
        self._output = ops.DivOp(c1, (ops.AddOp(c1, ops.ExpOp(
            ops.NegOp(self._input)))))

    def forward(self):
        return self._output.forward()

    def grad(self, partial_op=None):
        raise NotImplementedError


# alias for convenience
reduce_sum = _ReduceSum
sigmoid = _Sigmoid
