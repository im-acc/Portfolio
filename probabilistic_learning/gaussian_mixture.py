import numpy as np

DEFAULT_SIZE = 1000 # default sample size



class gaussian_mixture:
    '''
        Gaussian mixture :

            K : number of components
    '''
    def __init__(self, K=1): 
        self.K = K
        
        # paramaters initialisation
        self.pis = np.array(K*[1/K])
        self.mus = np.array(K*[None])
        self.sigmas = np.array(K*[None])

        # Density function
        self.p = lambda x : np.sum(self.pis*np.array([N_d(x, self.mus[:,k:k+1], self.sigmas[k]) for k in range(K)]))
    
    
    def sample(self, size = DEFAULT_SIZE, shuffle=True):
        '''
            Sampling from learned distribution
        '''
        nb_by_classes = [np.sum(np.random.choice(self.K, size, p=self.pis)==k) for k in range(self.K)]
        samples_ks_list = [np.random.multivariate_normal(self.mus[:,k], self.sigmas[k], nb_by_classes[k]).T
                            for k in range(self.K)]
        
        sample_x = np.concatenate(tuple(samples_ks_list), axis=1) 

        if shuffle:
            np.random.shuffle(sample_x)
        
        return sample_x
                            
                            
    def fit(self, X, nb_ite=50): 
        '''
            Paramater estimation via EM
        '''

        # X is an Dimension D x Nb of sample N matrix
        D = len(X)
        N = len(X[0])
        
        # Initialisation step
        self.mus = X[:,np.random.choice(N,self.K, False)]
        self.sigmas = [np.diag(np.var(X, axis=1)) for k in range(self.K)]
        
        # EM algorithm
        for i in range(nb_ite):
            
            # Expectation evaluation (E-step)
            R = np.zeros((self.K, N)) # K x N

            for n in range(N):
                xn = X[:, n:n+1]
                indiv_densities = np.array([ N_d(xn, self.mus[:,k:k+1], self.sigmas[k]) for k in range(self.K) ])
                comps = self.pis * indiv_densities
                comps_sum = np.sum(comps)
                R[:,n] = comps/comps_sum

            Nk = np.sum(R, axis=1).flatten()

            
            # Maximisation (M-step)

            # Covariances
            for k in range(self.K):
                self.sigmas[k] = 1/Nk[k]*np.sum(np.array([R[k,n]*(X[:,n:n+1]-self.mus[:,k:k+1])@
                                                 (X[:,n:n+1]-self.mus[:,k:k+1]).T for n in range(N)]), axis=0)
            
            # Mean :
            self.mus = X @ R.T @ np.diag(Nk**-1)
            
            # weights
            self.pis = Nk/N

def N_d(x, mu, sigma):
    '''
        Multivariate normal density function
    '''
    k = len(mu)
    d = x - mu
    return np.asscalar(((2*np.pi)**k*np.linalg.det(sigma))**-1/2*np.exp(-1/2*d.T @ np.linalg.inv(sigma) @ d))

mu1 = np.array([[3.0, 4.0]]).T
sigma1 = np.array([[0.3, 0.1],
                      [0.1, 0.3]])

x1 = np.random.multivariate_normal(np.array([1,1]), np.diag([0.3,0.1]), 500).T
x2 = np.random.multivariate_normal(mu1.flatten(), sigma1, 300).T
X_data = np.concatenate((x1,x2), axis=1)

gm1 = gaussian_mixture(K=2)
gm1.fit(X_data, nb_ite=50)

x_sample = gm1.sample(800)
