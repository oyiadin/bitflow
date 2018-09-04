import numpy as np
import matplotlib.pyplot as plt
import bitflow as tf


#### 生成用以分类的数据 ####


HOW_MANY = 80  # 数据点个数

true_w = np.random.rand() - 0.5
true_b = np.random.rand() - 0.5

train_XY = (np.random.rand(HOW_MANY, 2) - 0.5) * 2
train_Z = np.where((train_XY @ [[true_w], [-1]] + true_b) > 0, 1, 0)

print('generate successfully')


#### 开始训练 ####


with tf.Session() as sess:
    model = tf.models.LogisticRegression(units=(2, 1), learning_rate=0.001)
    model.fit(train_XY, train_Z, prompt_per_epochs=1, stop_when_delta=0.001)

    print('model fitted')

    pred = model.predict(train_XY)[:, 0]


#### 画图 ####


plt.scatter(train_XY[:, 0], train_XY[:, 1], c=np.where(pred > 0.5, 1, 0), label='prediction')
plt.plot(train_XY[:, 0], true_w * train_XY[:, 0] + true_b, label='real')

plt.show()
