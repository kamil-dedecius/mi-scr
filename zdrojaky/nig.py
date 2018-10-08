import numpy as np
from scipy.stats.distributions import t

class NiG():
    """Normal inverse-gamma prior"""
    
    def __init__(self, xi, nu):
        """Constructor"""
        self.xi = xi                             # hyperparameter xi
        self.nu = float(nu)                      # hyperparameter nu
        self.nparams = self.xi.shape[0] - 1.     # number of parameters
        self.a_log = []                          # log of parameter a
        self.b_log = []                          # log of parameter b
        self.V_log = []                          # log of scaling matrix
        self.Ebeta_log = []                      # log of E[beta]
        self.Esigma2_log = []                    # log of E[sigma2]
        self.var_beta_log = []                   # log of var(beta)
        self.var_sigma2_log = []                 # log of var(sigma2)
        self.beta_hat_log = []                   # for backward compatibility
        
    def update(self, y, X):
        """Bayesian update by observation y and regressor X"""
#        dt = np.r_[y, X]
#        self.xi += np.outer(dt, dt)
        if len(X.shape) > 1:
            dt = np.hstack((y[:,np.newaxis], X))
            prod = dt.T.dot(dt)
        elif len(X.shape) == 1 and np.isscalar(y):
            dt = np.insert(X, 0, y)
            prod = np.outer(dt, dt)
        self.xi += prod
        self.nu += np.size(y)
        
    def log(self):
        """Logging"""
        self.a_log.append(self.a)
        self.b_log.append(self.b)
        self.V_log.append(self.V)
        self.Ebeta_log.append(self.beta_hat)
        self.var_beta_log.append(self.beta_var)
        self.Esigma2_log.append(self.iG_mean)
        self.var_sigma2_log.append(self.iG_var)
        
    @property 
    def beta_hat(self):
        """Estimate of theta"""
        return np.dot(self.V, self.xi[1:, 0])
    Ebeta = beta_hat                             # Alias for compatibility
        
    @property
    def beta_var(self):
        """Variance of beta"""
        return np.diag(self.iG_mean * self.V)
    var_beta = beta_var                          # Alias for compatibility


    @property
    def iG_mean(self):
        """Estimate of variance sigma^2"""
        return self.b/(self.a - 1.)
    Esigma2 = iG_mean                            # Alias for compatibility

    @property
    def iG_var(self):
        """Variance of sigma2"""
        return self.b**2 / ((self.a - 1)**2 * (self.a - 2))
    var_sigma2 = iG_var                           # Alias for compatibility
        
    @property
    def a(self):
        """Hyperparameter a"""
        return .5 * self.nu
        
    @property    
    def b(self):
        """Hyperparameter b"""
        return (self.xi[0,0] - self.xi[0,1:].dot(self.V).dot(self.xi[0,1:].T)) / 2.
        
    @property
    def V(self):
        """Hyperparameter V (scaling matrix)"""
        return np.linalg.inv(self.xi[1:,1:])
        
    def predictive_logpdf(self, y, X):
        """Student predictive logpdf"""
        loc = X.dot(self.beta_hat)
        df = 2. * self.a
        tmp = X.dot(self.V).dot(X.T)
        if not np.isscalar(tmp):
            tmp = tmp.diagonal()
        scale = self.b/self.a * (1. + tmp)
        scale = np.sqrt(scale)
        return t.logpdf(y, df=df, loc=loc, scale=scale)        

#----------------------
if __name__ == '__main__':
    xi0 = np.diag([.01, .1, .1, .1])
    nu0 = 3.
    
    ndat = 300
    beta = np.array([.5, .1, -1.2])
    X = np.random.uniform(size=(ndat, 3))
    y = X.dot(beta) 
    y += np.random.normal(scale=0.1, size=y.shape)
    
    prior = NiG(xi0, nu0)
    sg2, sg2ab = [], []
    for yt, xt in zip(y, X):
        prior.update(yt, xt)
        print(prior.beta_hat)
#        print(prior.predictive_logpdf(y[:2], X[:2]))
    
        
