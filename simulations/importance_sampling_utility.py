import numpy as np

# Computing E[X | X ~ g] by importance sampling (sampling from f)
def E_imp_sampling(f_sampler, f, g, n=1000, normalized=True):
    '''
        f_sampler : vectorized sampler of X ~ f
        f : vectorized pmf/density function
        g : vectorized pmf/density function
    '''
    Xi = f_sampler(n)
    wi = g(Xi)/f(Xi) # weights
    estimate = Xi @ wi /n
    
    if normalized==False:
        estimate /=  wi.mean()
    
    return estimate

'''
# Test
f_sampler = np.random.randn # normal distribution
f = lambda x : np.exp(-(x**2)/2)/np.sqrt(2*np.pi)
g = lambda x : np.logical_and(x<1, x>0)  # uniform [0,1] distribution -> expectation of 1/2

# We expect values close to 1/2
E_imp_sampling(f_sampler, f, g, n=10**6)
'''
