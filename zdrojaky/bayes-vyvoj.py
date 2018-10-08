import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(14,5))
for i in range(1, 5):
    plt.subplot(1,4,i)
    plt.plot(x, norm.pdf(x, loc=np.random.normal(scale=.5), scale=1/i))
    plt.xlim(-5,5)
    plt.ylim(0, 1.6)
    plt.xlabel(r'$\theta$')
    if i == 1:
        plt.ylabel(r'$f(\theta)$')

plt.tight_layout()
plt.savefig('/tmp/bayes-evol.png')