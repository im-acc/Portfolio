import numpy as np

class Solver:
    
    # Jacobi method for solving 2D laplace equation by finite diffences
    # on [0,1] x [0,1] square with v(B)=0 {B : boundary}
    def laplace(fij, n=200, niter=50):
        v = np.zeros((n+2,n+2))
        h = 1/n
        for _ in range(niter):
            for i in range(1,n+1):
                for j in range(1,n+1):
                    v[i,j] = ( v[i-1,j] + v[i,j-1] + v[i+1,j] + v[i,j+1] - h**2 * f(i*h, j*h) ) / 4 
        return v

f = lambda x,y  : np.sin(np.pi*x)*np.sin(np.pi*y)/np.pi
#%%time
sol = Solver.laplace(f)
#plt.imshow(sol)
