import numpy as np
import matplotlib.pyplot as plt
import bitflow as tf


#### 生成用以拟合的数据 ####


HOW_MANY = 80  # 数据点个数

true_w = np.random.randn(1, 1) * 7
true_b = np.random.randn(1, 1) * 4

train_X = np.random.rand(HOW_MANY, 1) * 15
noise = np.random.randn(HOW_MANY, 1) * 5  # 随机噪音
train_Y = true_w * train_X + true_b + noise

print('generate successfully')


#### 开始训练 ####


with tf.Session() as sess:
    model = tf.models.LinearRegression(units=(1, 1), learning_rate=0.0000001)
    model.fit(train_X, train_Y, prompt_per_epochs=50, stop_when_delta=0.0000001)

    print('model fitted')

    _x = np.reshape(np.linspace(0, 15), (50, 1))
    _y = model.predict(_x)


#### 画图 ####


plt.plot(train_X, train_Y, 'o')
plt.plot(train_X, train_Y - noise, 'g', label='real')

plt.plot(_x, _y, 'r', label='prediction')

plt.legend()
plt.show()
