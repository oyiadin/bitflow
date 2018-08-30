# coding=utf-8

"""所有基础 Tensor 对象"""

import numpy as np
from . import session
from . import graph
from . import utils

__all__ = [
    'Constant', 'Placeholder', 'Variable',
    'AddOp', 'SubOp', 'MulOp', 'MatmulOp', 'DotOp', 'DivOp', 'PowOp'
]


class Tensor(object):
    """所有与计算直接相关的对象都以 Tensor 的形式储存，并提供相对统一的接口"""
    def __init__(self, name='Tensor', value=None):
        self.name = name
        self._real_value = value
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    # 为了与 Placeholder 的接口兼容 TAT
    @property
    def _value(self):
        return self._real_value

    @_value.setter
    def _value(self, value):
        self._real_value = value

    @property
    def shape(self) -> tuple:
        return np.shape(self._value)

    @property
    def dtype(self):  # not safe
        return self._value.dtype

    def forward(self):
        raise NotImplementedError

    def grad(self, partial_op=None):
        # 只支持标量式的自动求导
        raise NotImplementedError

    def __add__(self, rhs):
        return AddOp(self, rhs)

    def __radd__(self, lhs):
        return AddOp(lhs, self)

    def __sub__(self, rhs):
        return SubOp(self, rhs)

    def __rsub__(self, lhs):
        return SubOp(lhs, self)

    def __mul__(self, rhs):
        return MulOp(self, rhs)

    def __rmul__(self, lhs):
        return MulOp(lhs, self)

    def __matmul__(self, rhs):
        """Python 3.5 的新特性，a @ b 即表示 a.__matmul__(b)，矩阵乘法"""
        return MatmulOp(self, rhs)

    def __rmatmul__(self, lhs):
        return MatmulOp(lhs, self)

    def __div__(self, rhs):
        return DivOp(self, rhs)

    def __rdiv__(self, lhs):
        return DivOp(lhs, self)

    def __pow__(self, rhs):
        return PowOp(self, rhs)

    def __rpow__(self, lhs):
        return PowOp(lhs, self)

    def __str__(self):
        return "bf.Tensor({}_{})".format(self.name, hex(id(self))[2:])

    def __repr__(self):
        return "<bf.Tensor:{}_{}>".format(self.name, hex(id(self))[2:])


class Constant(Tensor):
    """常量，创建之后就不允许通过任何途径进行修改"""
    def __init__(self, value, dtype=None, name='constant'):
        super().__init__(name)
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=dtype)
        self._value = value

    def forward(self):
        return self._value

    def grad(self, partial_op=None):
        return 0


class Placeholder(Tensor):
    def __init__(self, dtype=None, shape=None, name='placeholder'):
        """shape 和 dtype 目前是两个没用的参数，仅为了后续的兼容性"""
        super().__init__(name)
        if shape:
            utils.make_sure_shape_valid(shape)

    @property
    def _value(self):
        return session.get_current_session().get_value(self)

    @_value.setter
    def _value(self, value):
        raise RuntimeError("cannot update value of a placeholder")

    def forward(self):
        if not session.get_current_session().is_fed(self):
            raise RuntimeError("all placeholders must be fed by `feed_dict`\n"
                               "  [note] when dealing with {}".format(self))
        ret = self._value
        return ret
        # 因为 self._value 是一个 property 对象，所以不能直接返回

    def grad(self, partial_op=None):
        return 0

    def feed(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if not utils.same_shape(value.shape, self.shape):
            raise ValueError(
                "your feed-in data for {} has a wrong `shape`".format(self))
        sess = session.get_current_session()
        sess.set_value(self, value)
        sess.add_to_fed(self)


class Variable(Constant):
    def __init__(self, value, name='Variable'):
        super().__init__(name=name, value=value)
        self._graph.add_to_trainable_variables_collection(self)

    def forward(self):
        return self._value

    def grad(self, partial_op=None):
        if partial_op == self:
            return 1
        else:
            return 0

    def set_value(self, value):
        """Variable 相对其他 Tensor 对象独有的接口，可以更新储存的值"""
        self._value = value

    def get_value(self):
        return self._value

# 梯度的计算止于上面三种对象
# 对于下面这些操作而言，梯度的计算则会不断往底层深入


class Operation(Tensor):
    def __init__(self, *objects, name='op'):
        """相对于 Tensor 来说，弱化了 value 的概念，多了对操作对象的预处理"""
        super().__init__(name)
        objects = list(objects)  # tuple is immutable
        for n, i in enumerate(objects):
            if not isinstance(i, Tensor):
                objects[n] = Constant(i)
                # 预处理，确保操作对象是 Tensor
                # 保持后续操作的统一性
        self._objs = objects
        if len(objects) == 2:  # binary op
            self._left = objects[0]
            self._right = objects[1]


class AddOp(Operation):
    def __init__(self, left, right, name='add'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() + self._right.forward()

    def grad(self, partial_op=None):
        return self._left.grad(partial_op) + self._right.grad(partial_op)


class SubOp(Operation):
    def __init__(self, left, right, name='sub'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() - self._right.forward()

    def grad(self, partial_op=None):
        return self._left.grad(partial_op) - self._right.grad(partial_op)


class MulOp(Operation):
    """逐元素乘法"""
    def __init__(self, left, right, name='mul'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() * self._right.forward()

    def grad(self, partial_op=None):
        return self._left.grad(partial_op) * self._right.forward() \
               + self._right.grad(partial_op) * self._left.forward()


class MatmulOp(Operation):
    """矩阵乘法"""
    def __init__(self, left, right, name='matmul'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() @ self._right.forward()

    def grad(self, partial_op=None):
        # 矩阵乘法太复杂了，链式法则不能用
        # 交给 models 和 nn 里人工计算结果写死在代码里吧…
        raise NotImplementedError


# alias
DotOp = MatmulOp


class DivOp(Operation):
    def __init__(self, left, right, name='div'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() / self._right.forward()

    def grad(self, partial_op=None):
        left_value = self._left.forward()
        left_grad = self._left.grad(partial_op)
        right_value = self._right.forward()
        right_grad = self._right.grad(partial_op)

        denominator = (left_grad * right_value - right_grad * left_value)
        return denominator / (right_value ** 2)


class PowOp(Operation):
    def __init__(self, left, right, name='pow'):
        super().__init__(left, right, name=name)
        if isinstance(right, np.ndarray) and right.shape != ():
            raise ArithmeticError('The right hand side object must be a scalar')
        if not isinstance(right, int):
            raise ValueError("the argument `power` must be a (int) scalar")

    def forward(self):
        return self._left.forward() ** self._right.forward()

    def grad(self, partial_op=None):
        power = self._right.forward()
        deeper_grad = self._left.grad(partial_op)
        power_grad = power * self._left.forward() ** (power - 1)
        return deeper_grad * power_grad
