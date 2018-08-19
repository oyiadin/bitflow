#coding=utf-8

from bitflow.nodes import tensor

__all__ = ['Add', 'Sub', 'Mul', 'MatMul', 'Div', 'Pow']


class Operation(object):
    '''所有操作的祖先，方便拓展新特性'''
    def __init__(self,  *objects):
        self.objects = objects

    def _calc(self, obj):
        '''对 obj 进行计算，并返回计算结果'''
        if not isinstance(obj, tensor.Tensor):
            raise ValueError(
                "Expects Tensor object but not {}".format(type(obj)))
        return obj.calc_fn()

    def gen_tensor(self):
        def calc_fn():
            left = self._calc(self.objects[0])
            for i in range(1, len(self.objects)):
                right = self._calc(self.objects[i])
                left = self.do_calc(left, right)
            return left
        return tensor.Tensor(name=self.name, calc_fn=calc_fn)

    def do_calc(self, *args, **kwargs):
        '''子类必须覆写此方法以实现计算'''
        raise NotImplementedError('This method must be overrided by subclass')


def check_if_any_non_tensor(func):
    def wrapper(self, *args, **kwargs):
        for i in range(len(args)):
            if not isinstance(args[i], tensor.Tensor):
                raise ValueError(
                    "Expects Tensor object but not {}".format(
                        type(args[i])))
        return func(self, *args, **kwargs)
    return wrapper


def check_argument_at_least(at_least):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if len(args) < at_least:
                raise ValueError(
                    "The operation requires at least 2 arguments")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


def check_argument_same_shape(func):
    def wrapper(self, *args, **kwargs):
        shape = args[0].shape
        for i in args:
            if i.shape != shape:
                raise ArithmeticError('All arguments must have the same shape')
        return func(self, *args, **kwargs)
    return wrapper


class Add(Operation):
    id_at = 0
    @check_if_any_non_tensor
    @check_argument_at_least(2)
    @check_argument_same_shape
    def __init__(self, *objects):
        self.name = 'Add:{}'.format(Add.id_at)
        super().__init__(*objects)
        Add.id_at += 1

    def do_calc(self, left, right):
        return left + right


class Sub(Operation):
    id_at = 0
    @check_if_any_non_tensor
    @check_argument_at_least(2)
    @check_argument_same_shape
    def __init__(self, *objects):
        self.name = 'Sub:{}'.format(Sub.id_at)
        super().__init__(*objects)
        Sub.id_at += 1

    def do_calc(self, left, right):
        return left - right


class Mul(Operation):
    id_at = 0
    @check_if_any_non_tensor
    @check_argument_at_least(2)
    def __init__(self, *objects):
        self.name = 'Mul:{}'.format(Mul.id_at)
        super().__init__(*objects)
        Mul.id_at += 1

    def do_calc(self, left, right):
        return left * right


class MatMul(Operation):
    '''TODO: shape check'''
    id_at = 0
    @check_if_any_non_tensor
    @check_argument_at_least(2)
    def __init__(self, *objects):
        self.name = 'MatMul:{}'.format(MatMul.id_at)
        super().__init__(*objects)
        MatMul.id_at += 1

    def do_calc(self, left, right):
        return left @ right


class Div(Operation):
    id_at = 0
    @check_if_any_non_tensor
    @check_argument_at_least(2)
    def __init__(self, *objects):
        self.name = 'Div:{}'.format(Div.id_at)
        super().__init__(*objects)
        Div.id_at += 1

    def do_calc(self, left, right):
        return left / right


class Pow(Operation):
    id_at = 0
    @check_if_any_non_tensor
    @check_argument_at_least(2)
    def __init__(self, *objects):
        self.name = 'Pow:{}'.format(Pow.id_at)
        super().__init__(*objects)
        Pow.id_at += 1

    def do_calc(self, left, right):
        if right.shape != ():
            raise ArithmeticError('The right hand side object must be a scalar')
        return left ** right
