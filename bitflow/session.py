# coding=utf-8

__all__ = ['Session', 'current_session', 'get_current_session']

current_session = None


class Session(object):
    """维护会话状态，只支持通过 with 语句块控制会话的进入与退出"""
    def __init__(self):
        if current_session:
            raise RuntimeError('there is already a Session instance')
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
        # Variable 的实例在每次计算后仍要保持好状态（值）
        # 会话期间持久存在的状态需要交由 Session 类进行维护
        return self._values.get(key)

    def set_value(self, key, value):
        self._values[key] = value

    def add_to_fed(self, tensor):
        """Placeholder 借此维护自己在当前会话中的投喂状态"""
        self._fed.append(tensor)

    def is_fed(self, tensor):
        """Placeholder.forward 在进行求值前都会借此方法确保自己已被投喂数据"""
        return tensor in self._fed

    def run(self, *tensors, feed_dict=None):
        self._fed = []
        # 每次 run 前都会清空，其实上次的数据还在，不过只认这个表
        for key in (feed_dict or {}):
            key.feed(feed_dict[key])

        ret = []
        if isinstance(tensors[0], (tuple, list)):
            tensors = tensors[0]
        for tensor in tensors:
            ret.append(tensor.forward())
        return ret if len(ret) != 1 else ret[0]


def get_current_session():
    if not current_session:
        raise RuntimeError('there is no Session instance yet\n'
                           '  [note] you must use the "with statement"')
    return current_session
