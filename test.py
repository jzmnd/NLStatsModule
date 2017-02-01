#! /usr/bin/env python

from NLStats import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def gen_data(t, a, b, c, noise=0, n_outliers=0, random_state=0):
	y = a + b * np.exp(t * c)

	rnd = np.random.RandomState(random_state)
	error = noise * rnd.randn(t.size)
	outliers = rnd.randint(0, t.size, n_outliers)
	error[outliers] *= 10

	return y + error

a = 0.5
b = 2.0
c = -0.8
x_min = 0
x_max = 10
n_points = 15

x_data = np.linspace(x_min, x_max, n_points)
y_data = gen_data(x_data, a, b, c, noise=0.1, n_outliers=2)

param0 = {'a':1., 'bb':1.5, 'cats012':-0, 'order':['a', 'bb', 'cats012']}

def residualsfunc(p, x, y):
	return p[0] + p[1] * np.exp(x * p[2]) - y

model1 = NLS(residualsfunc, param0, x_data, y_data, bounds=None, loss='soft_l1')

model1.fit()

model1.summary()

print model1.aic()

x_test = np.linspace(x_min, x_max, n_points * 10)
y_true = gen_data(x_test, a, b, c)
y_lsq = gen_data(x_test, *model1.parmEsts)

plt.plot(x_data, y_data, 'o')
plt.plot(x_test, y_true, 'k', linewidth=2, label='true')
plt.plot(x_test, y_lsq, label='fit')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
