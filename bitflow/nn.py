# coding=utf-8

# 其他代码对 nn 所有函数做的保证还没实现呢 XD
# 等我下一个 commit 吧～

import numpy as np
from . import ops

__all__ = ['reduce_sum']


class _ReduceSum(ops.Operation):
    def __init__(self, tensor, name='reduce_sum'):
        self._tensor = tensor
        super().__init__(name=name)

    def forward(self):
        return sum((i for i in self._tensor.forward()))

    def grad(self, partial_op=None):
        _temp = self._tensor.grad(partial_op)
        return _temp


class Raw(ops.Operation):
    def __init__(self, input_x, name='raw'):
        self._input_x = input_x
        super().__init__(name=name)

    def forward(self):
        return self._input_x.forward()

    def grad(self, partial_op=None):
        return np.ones_like(partial_op.forward())


# alias for convenience
reduce_sum = _ReduceSum
