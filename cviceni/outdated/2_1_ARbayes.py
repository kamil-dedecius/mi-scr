import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append('../zdrojaky')
from tsplot import tsplot
from nig import NiG
from scipy.io import loadmat

datafile = loadmat('/tmp/Gi.mat')
rolling_data = datafile['CDsel']

start = 300
end = 850
data = rolling_data[start:end,10]
ndat = data.size

tsplot(data)

ARorder = 2
xi0 = np.eye(ARorder+2) * 0.1
xi0[0,0] = 0.01
nu0 = 10

ar = NiG(xi0, nu0)

pred = []
for t in range(ndat):
    if t < ARorder or t == ndat:
        pred.append(0)
        continue
    
    data_for_pred = np.flip(data[t-ARorder+1:t+1], 0)
    pred.append(np.dot(ar.beta_hat, np.insert(data_for_pred, 0, 1)))
    
    regressor = np.insert(np.flip(data[t-ARorder:t],0), 0, 1)
    xt = data[t]
    ar.update(xt, regressor)
    ar.log()

#%%
Ebeta_log = np.array(ar.Ebeta_log)
varbeta_log = np.array(ar.var_beta_log)
Esigma2_log = np.array(ar.Esigma2_log)
varsigma2_log = np.array(ar.var_sigma2_log)
pred = np.array(pred)

#%%
plt.figure(figsize=(14, 2*ARorder))
for i in range(ARorder+1):
    plt.subplot(ARorder+1, 1, i+1)
    plt.plot(Ebeta_log[:,i])
    plt.plot(Ebeta_log[:,i] + 3*np.sqrt(varbeta_log[:,i]), 'gray')
    plt.plot(Ebeta_log[:,i] - 3*np.sqrt(varbeta_log[:,i]), 'gray')
plt.show()

#%%
plt.figure(figsize=(14,3))
plt.plot(Esigma2_log)
plt.plot(Esigma2_log - 3*np.sqrt(varsigma2_log))
plt.plot(Esigma2_log + 3*np.sqrt(varsigma2_log))
#%%

plt.figure(figsize=(14, 4))
plt.plot(data)
plt.plot(pred, '+')
plt.show()
#%%
residues = pred[ARorder+1:] - data[ARorder+1:]
print("RMSE: ", np.sqrt(np.mean(residues**2)))
plt.hist(residues, bins=15)