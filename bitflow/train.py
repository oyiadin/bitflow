#coding=utf-8

__all__ = ['GradientDescentOptimizer']

class Optimizer(object):
    '''优化器，用于降低代价函数。'''
    pass


class GradientDescentOptimizer(Optimizer):
    '''(随机)梯度下降，输入输出都是 Tensor 对象

    代价函数即为输入；
    生成的（输出）Tensor 对象中，`calc_fn` 函数会根据梯度修改各 Variable 对象的值'''
    pass