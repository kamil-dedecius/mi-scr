import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
import sys
sys.path.append('../zdrojaky')
from tsplot import tsplot

ndat = 100
x = np.zeros(ndat)
x[0] = .1
#x[1] = .5
epsilon = .01
beta = [.1, -.5, .4]
for t in range(1, ndat):
    x[t] = np.dot(beta, [1, x[t-1], x[t-2]]) + np.random.normal(scale=epsilon)
data = x
plt.figure()
tsplot(data)
plt.savefig('/tmp/l3-ar2.png', bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.scatter(data[i+1:], data[:-(i+1)])
    plt.title('Lag: {0}'.format(i+1))
    plt.axis('image')
plt.savefig('/tmp/l3-ar2-scatterplots.png', bbox_inches='tight')
#%%
model = AR(data)
result = model.fit(maxlag=2)
beta = result.params
print('beta', beta)
#%%
npreds = 100
for t in range(npreds):
    x_pred = np.dot(beta, [1, data[-1], data[-2]])
#    print(t, [1, data[-1]], x_pred)
    data = np.append(data, x_pred)
#%%
plt.figure(figsize=(14,4))
plt.plot(np.arange(ndat), data[:ndat], 'b')
plt.plot(np.arange(ndat, ndat+npreds), data[ndat:], 'r')
plt.savefig('/tmp/l3-ar2-pred.png', bbox_inches='tight')