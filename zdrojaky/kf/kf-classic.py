import numpy as np

class KF():
    def __init__(self, A, B, H, R, Q):
        self.A = A
        self.B = B
        self.H = H
        self.R = R
        self.Q = Q
        self.P = np.eye(self.Q.shape[0]) * 1000
        self.x = np.zeros(self.Q.shape[0]) + 10
        self.log_x = []
    
    def predict(self, u=None):
        xminus = self.A.dot(self.x)
        if u is not None:
            xminus += self.B.dot(u)
        Pminus = self.A.dot(self.P).dot(self.A.T) + self.Q
        self.x = xminus
        self.P = Pminus
    
    def correct(self, yt):
        HPHT = self.H.dot(self.P).dot(self.H.T)
        zavorka_inv = np.linalg.inv(HPHT + self.R)
        K = self.P.dot(self.H.T).dot(zavorka_inv)
        innovation = yt - self.H.dot(self.x)
        xplus = self.x + K.dot(innovation)
        zavorka = np.eye(K.shape[0]) - K.dot(self.H)
        KRKT = K.dot(self.R).dot(K.T)
        Pplus = (np.eye(4) - K.dot(self.H)).dot(self.P) #zavorka.dot(self.P).dot(zavorka) + KRKT
        self.x = xplus
        self.P = Pplus
    
    def log(self):
        self.log_x.append(self.x)
    
#%%--------------------------
if __name__ == '__main__':
    from trajectory import trajectory

    q = .5
    dt = 1.
    r = 3.
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    Q = q * np.array([[dt**3/3, 0      , dt**2/2, 0      ],
                      [0,       dt**3/3, 0,       dt**2/2],
                      [dt**2/2, 0,       dt,      0      ],
                      [0,       dt**2/2, 0,       dt     ]])
    H = np.array([[1., 0, 0, 0],
                  [0., 1, 0, 0]])
    R = r**2 * np.eye(2)
    
    code = 103
    traj = trajectory(code)
    
    kf = KF(A=A, B=None, H=H, R=R, Q=Q)
    for yt in traj.Y.T:
        kf.predict()
        kf.correct(yt)
        kf.log()
        
    plt.figure(figsize=(14,5))


    log_x = np.array(kf.log_x).T
    plt.plot(traj.Y[0,:], traj.Y[1,:], '.', label='Měření')
    plt.plot(log_x[0,:], log_x[1,:], '-', color='red', label='Filtrovaný odhad')
    plt.plot(traj.X[0,:], traj.X[1,:], 'k', label='Skutečnost')
    plt.grid(True)
    #plt.axis('equal')
    plt.legend()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.plot(log_x[2,:], 'r')
    plt.plot(traj.X[2,:], 'k')
    plt.subplot(1,2,2)
    plt.plot(log_x[3,:])
    plt.plot(traj.X[3,:], 'k')