## 为什么会有这个项目？

某天，我用 Tensorflow 的高阶 API 很轻松地就实现了一个 MLP 模型。但是，对我来说，它完全是一个黑盒，我只需要照着教程随便调一调参数就能获得很好的效果，这显然是不能让我满意的。抱着“实践是最好的老师”的想法，我想自己从零开始实现一次 MLP 模型。但是我发现，Tensorflow 项目实在是太复杂了，有 C++ 写成的部分，也有 Python 写成的部分，跳转满天飞，一不小心就会迷失在代码的海洋之中。我连 MLP 的梯度计算都找不到，更别说什么从零开始了。

我估摸了一下，计算图很好实现，自动求导也仅仅是个链式法则，矩阵计算的部分直接丢给 `numpy` 做苦力。所以……那就自己写一个 Tensorflow？花了半天把基本框架搭了出来，居然还真的跑起来了。

都写出来了，那就，继续做呗 XD

## 此项目的目标

Tensorflow 是本项目模仿的目标，但是不会保持与其的完全一致，替代 Tensorflow 那更是不可能的。

本项目不会用于生产环境，也不会考虑计算速度的快慢。对我而言，这个项目存在的意义有以下几点：

1. 提供一个更容易理解的 Tensorflow 实现
2. 提供一个便于亲手实现 DL/ML 模型、算法的框架

## 代码实例

### Tensorflow 中计算图、Session 部分的基础使用

```python
import bitflow as bf

constant2 = bf.constant(2, name='A Constant')
constant1 = bf.constant(1)
placeholder = bf.placeholder()
variable = bf.Variable(0)

result = constant2 * placeholder + constant1 + variable
print(constant2, constant1, result, sep='\n')

with bf.Session() as sess:
    print(sess.run(constant2))
    print(sess.run(result, feed_dict={placeholder: 3}))
    # sess.run(result)
    # 报错: 必须先对 placeholder 投喂数据

# 输出：
# bf.Tensor(A Constant:0_105bf2ef0)
# bf.Tensor(constant:0_105bf2eb8)
# bf.Tensor(add:1_106f9f588)
# 2
# 7
```
