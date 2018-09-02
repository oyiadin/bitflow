import time
import numpy as np
import matplotlib.pyplot as plt
import bitflow as bf


f = lambda x, y: x ** 2 + 5 * y ** 2


plt.ion()
plt_x = np.linspace(-0.5, 1.2, 40)
plt_y = np.linspace(-0.5, 1.2, 40)
plt_x, plt_y = np.meshgrid(plt_x, plt_y)
plt_z = f(plt_x, plt_y)
plt.contourf(plt_x, plt_y, plt_z, 25)

def paint(p1, p2):
    plt.scatter(*p1, color='red')
    plt.scatter(*p2, color='green')
    plt.pause(0.015)

paint((1, 1), (1, 1))
time.sleep(1.5)


with bf.Session() as sess:
    x1 = bf.Variable(1)
    y1 = bf.Variable(1)
    z1 = f(x1, y1)
    op1 = bf.train.GradientDescentOptimizer(0.0035).minimize(z1)

    x2 = bf.Variable(1)
    y2 = bf.Variable(1)
    z2 = f(x2, y2)
    op2 = bf.train.MomentumOptimizer(0.0035, 0.8).minimize(z2)

    for i in range(200):
        paint(sess.run(x1, y1), sess.run(x2, y2))
        sess.run(op1, op2)
