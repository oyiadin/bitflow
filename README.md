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

# Output:
# bf.Tensor(A Constant:0_105bf2ef0)
# bf.Tensor(constant:0_105bf2eb8)
# bf.Tensor(add:1_106f9f588)
# 2
# 7
```

### 标量的自动求导

```python
import bitflow as bf

x = bf.Variable(2)
y = bf.Variable(1)
z = (x * y + x - 1) ** 2 + y

with bf.Session() as sess:
    print('z =', sess.run(z))
    print('∂z/∂x =', z.grad(x))
    print('∂z/∂y =', z.grad(y))

# z = (2 * 1 + 2 - 1) ** 2 + 1
#   = 10
# ∂z/∂x = 2 * (x * y + x - 1) * (y + 1)
#       = 2 * (2 * 1 + 2 - 1) * (1 + 1)
#       = 12
# ∂z/∂y = 2 * (x * y + x - 1) * x + 1
#       = 2 * (2 * 1 + 2 - 1) * 2 + 1
#       = 13

# Output:
# z = 10
# ∂z/∂x = 12
# ∂z/∂y = 13
```

### 线性回归的标准实现 (标量)

```python
import numpy as np
import bitflow as bf

EPOCHS = 50
LEARNING_RATE = 0.003
HOW_MANY_POINTS = 100

# generate some random points
true_w = int(np.random.randn() * 5)
true_b = int(np.random.randn() * 5)

train_x = np.random.randn(HOW_MANY_POINTS)
noise = np.random.randn(HOW_MANY_POINTS)  # random noises
train_y = true_w * train_x + true_b + noise

# create some necessary tensors
x = bf.placeholder()  # sample
y = bf.placeholder()  # label
w = bf.Variable(np.random.rand())
b = bf.Variable(np.random.rand())
pred = x * w + b
loss = bf.nn.reduce_sum((pred - y) ** 2)
optimizer = bf.train.GradientDescentOptimizer(
    learning_rate=LEARNING_RATE).minimize(loss)

# train our linear regression model
with bf.Session() as sess:
    for epoch in range(1, EPOCHS):
        for _x, _y in zip(train_x, train_y):
            sess.run(optimizer, feed_dict={x: _x, y: _y})
        if not epoch % 1:
            print('Epoch #{}, loss={}, w={}, b={}'.format(epoch,
                *sess.run(loss, w, b, feed_dict={x: train_x, y: train_y})))

    print('model trained successfully')
    print('final value: w = {}, b = {}'.format(*sess.run(w, b)))
    print('while the true_w = {}, true_b = {}'.format(true_w, true_b))

# Output:
# Epoch #1, loss=160.02968143915703, w=0.9578589353193588, b=1.149858066725097
# Epoch #2, loss=116.6527173429976, w=1.0152174988683842, b=1.5060229892078008
# Epoch #3, loss=104.4700270394287, w=1.0280963017191411, b=1.6986649076737153
# Epoch #4, loss=100.95945166619198, w=1.0251595318310058, b=1.8037958297945236
# ......
# Epoch #46, loss=99.47348856016569, w=0.9946397244743984, b=1.935687261742318
# Epoch #47, loss=99.47348856014511, w=0.9946397244597486, b=1.935687261757443
# Epoch #48, loss=99.47348856013275, w=0.9946397244509715, b=1.9356872617665033
# Epoch #49, loss=99.47348856012535, w=0.9946397244457121, b=1.9356872617719318
# model trained successfully
# final value: w = 0.9946397244457121, b = 1.9356872617719318
# while the true_w = 1, true_b = 2
```