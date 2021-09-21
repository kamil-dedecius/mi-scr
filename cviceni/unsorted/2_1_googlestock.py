import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
import sys
sys.path.append('../zdrojaky')
from tsplot import tsplot


data = np.genfromtxt('googlestock.txt', skip_header=1, usecols=1, autostrip=True)
ndat = data.size
tsplot(data)
plt.figure(figsize=(6,6))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.scatter(data[i+1:], data[:-(i+1)])
    plt.title('Lag: {0}'.format(i+1))
    plt.axis('image')

#%%
model = AR(data)
result = model.fit(maxlag=1)
beta = result.params

#%%
npreds = 100
for t in range(npreds):
    x_pred = np.dot(beta, [1, data[-1]])
    print(t, [1, data[-1]], x_pred)
    data = np.append(data, x_pred)
#%%
plt.plot(np.arange(ndat), data[:ndat], 'b')
plt.plot(np.arange(1,ndat), result.predict(), 'g')
plt.plot(np.arange(ndat, ndat+npreds), data[ndat:], 'r')

#%%
plt.hist(result.resid)

#%%
data_pred = []
for t in range(1, ndat):
    data_pred.append(np.dot(beta, [1, data[t-1]]))
    
plt.plot(np.arange(ndat), data[:ndat])
plt.plot(np.arange(1,ndat), data_pred)