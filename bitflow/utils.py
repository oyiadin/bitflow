# coding=utf-8

# shape 方面基本还没开工，各种 broadcast 也没做，慢慢来吧 qwq


def make_sure_shape_valid(shape):
    if not isinstance(shape, (tuple, list)):
        raise ValueError("`shape` must be a tuple or a list")

    def checker(inner):
        for i in inner:
            if isinstance(i, (tuple, list)):
                if not checker(i):
                    return False
            elif not isinstance(i, int):
                return False
        return True

    if not checker(shape):
        raise ValueError("`shape` must be a tuple/list containing integers")


def same_shape(*shapes):
    return True  # TODO


def exactly_same_shape(*shapes):
    shape = shapes[0]
    for i in shape:
        if i != shape:
            return False
    return True
