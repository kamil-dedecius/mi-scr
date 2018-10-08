import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table


height = np.load('/tmp/tracking.npz')['y']
time = np.arange(y.size)
data = {'height': height, 'time': time}
regression = smf.ols('height ~ time', data=data).fit()
st, dt, ss2 = summary_table(regression, alpha=0.05)
fittedvalues = dt[:,2]
predict_ci_low, predict_ci_upp = dt[:,6:8].T
predict_mean_se  = dt[:,3]

plt.figure(figsize=(14,5))
plt.subplot(121)
plt.plot(y, '+', label="data")
plt.plot(predict_ci_low, '--r')
plt.plot(predict_ci_upp, '--r')
plt.plot(fittedvalues, 'g')
plt.subplot(122)
plt.hist(regression.resid, bins=15)
plt.savefig('/tmp/tracking-plot.png')
print(regression.summary())

#%%
plt.figure(2)
plt.scatter(time, height, marker='+')
plt.savefig('/tmp/tracking-scatterplot.png')