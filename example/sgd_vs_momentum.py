import time
import numpy as np
import matplotlib.pyplot as plt
import bitflow as bf


plt.ion()
plt_x = np.linspace(0, 1.5, 20)
plt_y = np.linspace(0, 2, 20)
plt_x, plt_y = np.meshgrid(plt_x, plt_y)

height = lambda x, y: (x - 2) ** 2 + 2 * (y - 2) ** 2 + x * y


def paint(p1, p2):
    plt.contourf(plt_x, plt_y, height(plt_x, plt_y), 6)
    plt.scatter(*p1, color='red', marker='o')
    plt.scatter(*p2, color='green', marker='x')
    plt.pause(0.015)

paint((0.5, 0.5), (0.5, 0.5))
time.sleep(1)


with bf.Session() as sess:
    x1 = bf.Variable(0.5)
    y1 = bf.Variable(0.5)
    z1 = (x1 - 2) ** 2 + 2 * (y1 - 2) ** 2 + x1 * y1
    op1 = bf.train.GradientDescentOptimizer(0.0035).minimize(z1)

    x2 = bf.Variable(0.5)
    y2 = bf.Variable(0.5)
    z2 = (x2 - 2) ** 2 + 2 * (y2 - 2) ** 2 + x2 * y2
    op2 = bf.train.MomentumOptimizer(0.0035, 0.85).minimize(z2)

    for i in range(200):
        paint(sess.run(x1, y1), sess.run(x2, y2))
        sess.run(op1, op2)
