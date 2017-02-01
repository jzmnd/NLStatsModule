#! /usr/bin/env python

from NLStats import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

sold_data = np.array([141, 166, 161, 170, 148, 136, 169, 109, 117, 87, 105, 73, 82, 75])
cord_data = np.array([6.4, 6.1, 5.7, 6.9, 7.0 ,7.2, 6.6, 5.7, 5.7, 5.3, 4.9, 5.4, 4.5, 6.0])

param0 = {'const':0., 'slope':0, 'order':['const', 'slope']}

def residualsfunc(p, x, y):
	return p[0] + p[1] * x - y

model1 = NLS(residualsfunc, param0, sold_data, cord_data, bounds=None, loss='linear')

model1.fit()

model1.summary()

print model1.aic()

x_test = np.linspace(60, 180, 60)
y_lsq = model1.parmEsts[0] + x_test * model1.parmEsts[1]

plt.plot(sold_data, cord_data, 'o')
plt.plot(x_test, y_lsq, label='fit')
plt.xlabel("sold")
plt.ylabel("cord")
plt.legend()
plt.show()
