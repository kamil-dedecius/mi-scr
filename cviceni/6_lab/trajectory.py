import numpy as np
from scipy.stats import multivariate_normal as mvn

class trajectory():
    
    def __init__(self, seed=123, ndat=100, q=.5, r=3.):
        self.ndat = ndat
        self.seed = seed
        self.q = q
        self.dt = 1
        self.r = r
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.Q = self.q**2 * np.array([[self.dt**3/3, 0      , self.dt**2/2, 0      ],
                                      [0,       self.dt**3/3, 0,       self.dt**2/2],
                                      [self.dt**2/2, 0,       self.dt,      0      ],
                                      [0,       self.dt**2/2, 0,       self.dt     ]])
        #self.Q = self.q**2 * np.eye(4)
        self.H = np.array([[1., 0, 0, 0],
                           [0., 1, 0, 0]])
        self.R = self.r**2 * np.eye(2)
        self.m0 = np.array([0., 0., 1., -1.]) #.reshape(4,1)
        self.X = np.zeros(shape=(self.A.shape[0], self.ndat))
        self.Y = np.zeros(shape=(self.H.shape[0], self.ndat))
        self._simulate()
        
    def _simulate(self):
        np.random.seed(self.seed)
        
        x = self.m0;
        for t in range(self.ndat):
            w = mvn.rvs(cov=self.Q)
            x = self.A.dot(x) + w
            y = self.H.dot(x) + mvn.rvs(cov=self.R)
            self.X[:,t] = x.flatten()
            self.Y[:,t] = y.flatten()


if __name__ == '__main__':
    import matplotlib.pylab as plt
    traj = trajectory(12345)
    
    plt.figure(1)
    plt.plot(traj.Y[0,:], traj.Y[1,:], '+')
    plt.plot(traj.X[0,:], traj.X[1,:])

    plt.figure(2)
    plt.subplot(1,2,1)
    plt.plot(traj.X[2,:])    
    plt.subplot(1,2,2)
    plt.plot(traj.X[3,:])
