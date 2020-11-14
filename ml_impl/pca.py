import numpy as np

class pca:  
    '''
    Return P, var(Yi), comp_max first components
    '''
    def decomposition(X, comp_max = None):  # X : designe matrix (N x D)
        N = len(X)
        D = len(X[0])
        
        if (comp_max == None):
            comp_max = D
        
        # Center X
        X_c = X - np.mean(X, axis=0)
        
        # Covariance Matrix of X : cov(X)
        covX = 1/N * X_c.T @ X_c
        l, P = np.linalg.eig(covX)
        
        indices = np.argsort(l)[::-1][:comp_max]
        l = l[indices]
        P = P[:,indices]
        
        return l, P

    def dim_reduction(X, nb_dim=None):
        if nb_dim==None:
            nb_dim = len(X[0])
            
        l, P = pca.decomposition(X, comp_max = nb_dim)
        return X @ P    
    def project(X, nb_dim=None): # Project the reduced low dimensional data to the original space
        if nb_dim==None:
            nb_dim = len(X[0])
        
        mu = np.mean(X, axis=0)
        
        l, P = pca.decomposition(X, comp_max = nb_dim)
        return (X-mu) @ P @ P.T + mu
