#coding=utf-8

__all__ = ['Session', 'get_current_session', 'current_session']

current_session = None

class Session(object):
    def __init__(self):
        self._values = {}
        self._fed = []

    def __enter__(self):
        global current_session
        current_session = self
        return self

    def __exit__(self, type, value, trace):
        global current_session
        current_session = None

    def get_value(self, key):
        return self._values.get(key)

    def set_value(self, key, value):
        self._values[key] = value

    def add_to_fed(self, tensor):
        self._fed.append(tensor)

    def is_fed(self, tensor):
        return tensor in self._fed

    def run(self, *tensors, feed_dict={}):
        self._fed = []
        for key in feed_dict:
            key.feed(feed_dict[key])

        ret = []
        for tensor in tensors:
            ret.append(tensor.forward())
        return ret if len(ret) != 1 else ret[0]


def get_current_session():
    if not current_session:
        raise RuntimeError('there is no Session instance yet\n'
                           '[note] only works correctly with "with statement"')
    return current_session
