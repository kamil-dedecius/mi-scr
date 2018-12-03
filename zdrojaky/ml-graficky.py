import numpy as np
from scipy.stats import norm
import matplotlib.pylab as plt

xx = np.linspace(2,8, 100)
Y = np.array([5.1, 4.8, 4.6, 5, 5.4])
plt.figure(figsize=(14,4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(xx, norm.pdf(xx, loc=4+i, scale=1), color='darkorange')
    plt.stem(Y, norm.pdf(Y, loc=4+i, scale=1), basefmt=' ')
    plt.grid(None)
    fy = np.prod(norm.pdf(Y, loc=4+i, scale=1))
    plt.title(r'$f(y_1,\ldots,y_5|a={0:1}) = {1:.2g}$'.format(4+i, fy, np.log10(fy)))
    
plt.savefig('/tmp/ml-graficky.png', bbox_inches='tight')
    