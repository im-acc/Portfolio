import torch
import numpy as np

'''
Labeled data dimensionality reduction N-dim |--> 1-dim
'''
class lda:
    
    def __init__(self, X_d):
        # X : Nb Data x (Nb features + 1) last col : classes  
        
        self.classes = torch.tensor(np.unique(X[:,-1]))
        self.nbC = torch.tensor([(X[:,-1]==c).sum() for c in self.classes])
        
        self.N = len(X) # Nb data
        self.D = len(X[0]) - 1 # Nb features
        self.w = torch.randn(self.D, 1, dtype=torch.float64, requires_grad = True)
        
    def train(self, lr=1e-4, nb_epoches=10):
        lossList = []
        for i in range(nb_epoches):
            # Forward Error pass

            # Spread within classes
            Sw = sum([ self.nbC[c]*LDA.cov(X[X[:,-1]==c][:,:-1]) for c in range(len(self.classes)) ])

            # Sb calculation, spread between classes (center's)
            Sb = torch.zeros(self.D, self.D, dtype=torch.float64)
            for c in range(len(self.classes)):
                mu_c = X[X[:,-1]==c].mean(axis=0)[:-1].reshape(self.D, 1)
                mu = X[:,:-1].mean(axis=0).reshape(self.D, 1)
                Sb += (mu_c-mu) @ (mu_c-mu).T


            loss = ((self.w.T @ Sw @ self.w) / (self.w.T @ Sb @ self.w) )
            
            # Backward and weights updates
            loss.backward()
            self.w.data -= lr*self.w.grad

            # Initialise the gradiant
            self.w.grad.zero_()
            
            lossList.append(loss)
            
        self.w = self.w.detach()
        return lossList
        
    def cov(X):
        n = len(X)
        mu = X.mean(axis=0)
        return 1/n*sum([ (X[k:k+1]-mu).T @ (1 + X[k:k+1]-mu) for k in range(n) ])
    
    def nw(self):
        self.w.data /= torch.norm(self.w.data, 2)
    
    def proj(self, X):
        self.nw()
        w = self.w.data
        return (w @ w.T @ X.T).T
    
    def dim_reduction(self, X):  # --> 1-D, coordinates of the projection: x_proj = 
        self.nw()
        w = self.w.data
        return X @ w
    
    def dim_augment(self, c):
        return c @ self.w.data.T
