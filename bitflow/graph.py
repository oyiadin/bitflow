# coding=utf-8

__all__ = ['Graph', 'default_graph', 'get_default_graph', 'set_default_graph']
default_graph = None


class Graph(object):
    """Graph 实例被用来维护几张有用的表，以及做好命名的序号安排"""
    def __init__(self):
        self._variables = {}
        self._trainable_variables_collection = []
        self._id_for_unique_name = {}

    def _get_unique_name(self, name: str):
        """对给定名字标上独一无二的序号"""
        self._id_for_unique_name[name] = \
            self._id_for_unique_name.get(name, -1) + 1
        return '{}:{}'.format(name, self._id_for_unique_name[name])

    def add_to_graph(self, tensor):
        """将 Tensor 对象添加到本实例中，并赋予其唯一的命名"""
        tensor.name = self._get_unique_name(tensor.name)
        self._variables[tensor.name] = tensor

    def get_trainable_variables_collection(self):
        return self._trainable_variables_collection

    def add_to_trainable_variables_collection(self, variable):
        self._trainable_variables_collection.append(variable)


def get_default_graph():
    """若尚无 Graph 实例，则创建并返回之；否则返回已有实例

    与 Session 的行为不同，因为我觉得一般没有切换 Graph 实例的必要"""
    global default_graph
    if not default_graph:
        default_graph = Graph()
    return default_graph


def set_default_graph(graph):
    global default_graph
    default_graph = graph
    return graph
