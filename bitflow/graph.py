#coding=utf-8

__all__ = ['Graph', 'get_default_graph', 'set_default_graph', 'default_graph']
default_graph = None

class Graph(object):
    def __init__(self):
        self._variables = {}
        self._trainable_variables_collection = []
        self._id_for_unique_name = {}

    def _get_unique_name(self, name):
        self._id_for_unique_name[name] = self._id_for_unique_name.get(name, -1) + 1
        return '{}:{}'.format(name, self._id_for_unique_name[name])

    def add_to_graph(self, tensor):
        tensor.name = self._get_unique_name(tensor.name)
        self._variables[tensor.name] = tensor

    def get_trainable_variables_collection(self):
        return self._trainable_variables_collection

    def add_to_trainable_variables_collection(self, variable):
        self._trainable_variables_collection.append(variable)


def get_default_graph():
    global default_graph
    if not default_graph:
        default_graph = Graph()
    return default_graph


def set_default_graph(graph):
    global default_graph
    default_graph = graph
    return graph
