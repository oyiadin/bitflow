#coding=utf-8

import numpy as np
from bitflow.nodes import ops

class Tensor(object):
    '''bitflow 里所有数据都以 Tensor 的形式储存，并提供相对统一的接口。

    Tensor 类抽象了“数据”这一概念，保存着“原始数据”及得到最终数据的映射(函数)。对于常量对象，此函数直接返回原始数据。

    Operation 类在实例化时，接受 Tensor 对象作为输出，并创建特定 Tensor，自动设定好映射的函数'''
    def __init__(self, array=None, shape=None, dtype=None, name=None,
                 calc_fn=None):
        if array and not calc_fn and not isinstance(array, np.ndarray):
            array = np.array(array, dtype=dtype)

        if not shape is None:
            if not (isinstance(shape, list) or isinstance(shape, tuple)):
                raise ValueError("Argument `shape` must be a list or a tuple")
            shape = tuple(shape)
            if isinstance(array, np.ndarray) and array.shape != shape:
                raise ValueError(
                    "The shape of `array` didn't fit with the argument `shape`")
        # 若有传入 array 而没有传入 shape
        elif isinstance(array, np.ndarray):
            shape = array.shape

        self.array = array
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.calc_fn = calc_fn or (lambda: self.array)
        super().__init__()

    def __add__(self, rhs):
        return ops.Add(self, rhs).gen_tensor()

    def __sub__(self, rhs):
        return ops.Sub(self, rhs).gen_tensor()

    def __mul__(self, rhs):
        return ops.Mul(self, rhs).gen_tensor()

    def __matmul__(self, rhs):
        '''Python 3.5 的新特性，a @ b 即表示 a.__matmul__(b)，矩阵乘法'''
        return ops.MatMul(self, rhs).gen_tensor()

    def __div__(self, rhs):
        return ops.Div(self, rhs).gen_tensor()

    def __pow__(self, rhs):
        return ops.Pow(self, rhs).gen_tensor()

    def __str__(self):
        return "bf.Tensor(`{}`, shape={}, dtype={}) at {}".format(
            self.name, self.shape, self.dtype, hex(id(self)))

    def __repr__(self):
        return "<bf.Tensor `{}`, shape={}, dtype={}, at {}>".format(
            self.name, self.shape, self.dtype, hex(id(self)))


class constant(Tensor):
    id_at = 0
    def __init__(self, array, *args, **kwargs):
        super().__init__(array, name='constant:{}'.format(constant.id_at),
                         *args, **kwargs)
        constant.id_at += 1


class placeholder(Tensor):
    id_at = 0
    def __init__(self, **kwargs):
        def calc_fn():
            '''placeholder 需要在 run() 时投喂数据，就是在这个函数检查的'''
            if not self.array:
                raise ValueError(
                    "A placeholder must be fed before do any calculations")
            return self.array
        super().__init__(name='placeholder:{}'.format(placeholder.id_at),
                         calc_fn=calc_fn, **kwargs)
        placeholder.id_at += 1


class Variable(Tensor):
    id_at = 0
    def __init__(self, **kwargs):
        super().__init__(name='Variable:{}'.format(Variable.id_at),
                         calc_fn=calc_fn, **kwargs)
        Variable.id_at += 1

