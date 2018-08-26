#coding=utf-8

'''所有对象都是 Tensor 的子类 LOL'''

import numpy as np
from . import session
from . import graph
from . import utils

__all__ = [
    'constant', 'placeholder', 'Variable', 
    'AddOp', 'SubOp', 'MulOp', 'MatmulOp', 'DivOp', 'PowOp'
]


class Tensor(object):
    '''所有与计算直接相关的对象都以 Tensor 的形式储存，并提供相对统一的接口'''
    def __init__(self, name='Tensor'):
        self.name = name
        self._graph = graph.get_default_graph()
        self._graph.add_to_graph(self)

    def forward(self):
        raise NotImplementedError

    def grad(self, partial_derivative_op=None):
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
        '''Python 3.5 的新特性，a @ b 即表示 a.__matmul__(b)，矩阵乘法'''
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


class constant(Tensor):
    def __init__(self, value, dtype=None, name='constant'):
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=dtype)

        self._value = value
        self._shape = value.shape
        self._dtype = dtype
        super().__init__(name)

    def forward(self):
        return self._value

    def grad(self, partial_derivative_op=None):
        return 0


class placeholder(Tensor):
    def __init__(self, shape=None, dtype=None, name='placeholder'):
        if shape:
            utils.make_sure_shape_valid(shape)
        self._shape = shape
        self._dtype = dtype
        super().__init__(name)

    @property
    def _value(self):
        return session.get_current_session().get_value(self)

    @property
    def _is_fed(self):
        return session.get_current_session().is_fed(self)

    def forward(self):
        if not self._is_fed:
            raise RuntimeError("all placeholders must be fed by `feed_dict`\n"
                               "[note] when dealing with {}".format(self))
        ret = self._value
        return ret

    def grad(self, partial_derivative_op=None):
        return 0

    def feed(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if not utils.same_shape(value.shape, self._shape):
            raise ValueError(
                "your feed-in data for {} has a wrong `shape`".format(self))
        sess = session.get_current_session()
        sess.set_value(self, value)
        sess.add_to_fed(self)


class Variable(constant):
    def __init__(self, value, name='Variable', **kwargs):
        super().__init__(value, **kwargs, name=name)
        graph.get_default_graph().add_to_trainable_variables_collection(self)

    def forward(self):
        return self._value

    def grad(self, partial_derivative_op=None):
        if self == partial_derivative_op:
            return 1
        else:
            return 0

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value


class Operation(Tensor):
    def __init__(self, *objects, name='op'):
        objects = list(objects)
        for n, i in enumerate(objects):
            if not isinstance(i, Tensor):
                objects[n] = constant(i)
        self._objects = objects
        if len(objects) == 2:  # binary op
            self._left = objects[0]
            self._right = objects[1]
        super().__init__(name)


class AddOp(Operation):
    def __init__(self, left, right, name='add'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() + self._right.forward()

    def grad(self, partial_derivative_op=None):
        return self._left.grad(partial_derivative_op) \
               + self._right.grad(partial_derivative_op)


class SubOp(Operation):
    def __init__(self, left, right, name='sub'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() - self._right.forward()

    def grad(self, partial_derivative_op=None):
        return self._left.grad(partial_derivative_op) \
               - self._right.grad(partial_derivative_op)


class MulOp(Operation):
    '''逐元素乘法'''
    def __init__(self, left, right, name='mul'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() * self._right.forward()

    def grad(self, partial_derivative_op=None):
        return self._left.grad(partial_derivative_op) * self._right.forward() \
               + self._right.grad(partial_derivative_op) * self._left.forward()


class MatmulOp(Operation):
    '''矩阵乘法 TODO: shape check'''
    def __init__(self, left, right, name='matmul'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() @ self._right.forward()

    def grad(self, partial_derivative_op=None):
        raise NotImplementedError(
            "operation matmul doesn't support gradient calculating right now")


class DivOp(Operation):
    def __init__(self, left, right, name='div'):
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() / self._right.forward()

    def grad(self, partial_derivative_op=None):
        return (self._left.grad(partial_derivative_op) * self._right.forward() \
                - self._right.grad(partial_derivative_op) * self._left.forward()) \
               / self._right.forward() ** 2


class PowOp(Operation):
    def __init__(self, left, right, name='pow'):
        if isinstance(right, np.ndarray) and right.shape != ():
            raise ArithmeticError('The right hand side object must be a scalar')
        if not isinstance(right, int):
            raise ValueError("the argument `power` must be a (int) scalar")
        super().__init__(left, right, name=name)

    def forward(self):
        return self._left.forward() ** self._right.forward()

    def grad(self, partial_derivative_op=None):
        return self._right.forward() * (
                self._left.forward() ** (self._right.forward() - 1)) \
               * self._left.grad(partial_derivative_op)