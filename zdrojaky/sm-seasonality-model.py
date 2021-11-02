import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

class SeasonalityModel(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Model order
        k_states = k_posdef = 12

        # Initialize the statespace
        super(SeasonalityModel, self).__init__(
            endog, k_states=k_states, k_posdef=k_posdef,
            initialization='approximate_diffuse',
            loglikelihood_burn=k_states
        )

        # Initialize the matrices
        self.ssm['design'] = np.zeros(k_states); self.ssm['design'][0,0] = 1
        trn = np.c_[np.eye(k_states-1), np.zeros((k_states-1,1))]
        trn = np.r_[-1*np.ones((1,k_states)), trn]
        self.ssm['transition'] = trn
        self.ssm['selection', 0, 0] = 1
        

    @property
    def param_names(self):
        return ['sigma2.process']

    @property
    def start_params(self):
        return [np.std(self.endog)]*2

    def transform_params(self, unconstrained):
        return unconstrained**2

    def untransform_params(self, constrained):
        return constrained**0.5

    def update(self, params, *args, **kwargs):
        params = super(SeasonalityModel, self).update(params, *args, **kwargs)
        
        # Observation covariance
        self.ssm['state_cov',0,0] = params[0]


#%% Simulace
if __name__ == '__main__':
    np.random.seed(12)
    ndat = 100
    data = np.zeros(10)
    mod = SeasonalityModel(data)
    initstate = np.random.uniform(-1, 1, size=12)
    initstate[-1] = -np.sum(initstate[:11])
    Y = mod.simulate(params=1.6, nsimulations=ndat, initial_state=initstate)
    plt.plot(Y)

    model = SeasonalityModel(Y)
    res = model.fit(start_params=0)
    res.summary()

#%%
    nsteps = 30
    ndat = Y.size
    predict = res.get_prediction()
    forecast = res.get_forecast(nsteps)

    plt.figure(figsize=(10,4))
    plt.plot(Y, label='Y')
    plt.plot(predict.predicted_mean, label='Predicted')
    plt.plot(np.arange(ndat, ndat+nsteps), forecast.predicted_mean, label='Forecast')
    forecast_ci = forecast.conf_int()
    forecast_index = np.arange(ndat, ndat + nsteps)
#plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='green', alpha=0.1)
    plt.legend()
