import numpy as np
import matplotlib.pylab as plt
from nig import NiG

y = np.load('tracking.npz')['y']

xi0 = np.diag([10000, .1, .1])   # Prior xi_0
nu0 = 5.                         # Prior nu_0
regmodel = NiG(xi0, nu0)         # NiG object

for t in range(ndat):
    yt = y[t]
    xt = np.array([1, (t+1)**2])
    
    regmodel.update(yt, xt)      # update of the prior
    regmodel.log()               # logging

print('Final estimate of beta: ', regmodel.Ebeta)
print('Final estimate of sigma2: ', regmodel.Esigma2)
print('std(beta): ', np.sqrt(regmodel.var_beta))
print('std(sigma2): ', np.sqrt(regmodel.var_sigma2))

Ebeta_log = np.array(regmodel.Ebeta_log)
std_beta_log = np.array(regmodel.var_beta_log)

plt.figure(figsize=(15,8))
plt.subplot(311)
plt.plot(Ebeta_log[:, 0])
plt.fill_between(np.arange(ndat),
                 Ebeta_log[:, 0] + 1 * std_beta_log[:, 0],
                 Ebeta_log[:, 0] - 1 * std_beta_log[:, 0],
                 color='red'
                 )
plt.ylim(-300, 300)
plt.ylabel(r'$\beta_0$')

plt.subplot(312)
plt.plot(Ebeta_log[:, 1])
plt.fill_between(np.arange(ndat),
                 Ebeta_log[:, 1] + 1 * std_beta_log[:, 1],
                 Ebeta_log[:, 1] - 1 * std_beta_log[:, 1],
                 color='red'
                 )
plt.ylim(-100, 100)
plt.ylabel(r'$\beta_1$')

std_sigma2_log = np.sqrt(np.array(regmodel.var_sigma2_log))

plt.subplot(3,1,3)
plt.plot(regmodel.Esigma2_log)
plt.fill_between(np.arange(ndat),
                 np.array(regmodel.Esigma2_log) + 1. * std_sigma2_log,
                 np.array(regmodel.Esigma2_log) - 1. * std_sigma2_log,
                 color='red'
                 )
plt.ylim(-2500, 2500)
plt.ylabel(r'$\sigma^2$')

plt.savefig('/tmp/tracking-bayes.png')