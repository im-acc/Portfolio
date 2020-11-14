import numpy as np

#functions
def loss(a, y): #a : output, y : label
    d = y-a
    return np.asscalar(0.5*d.T@d)

def d_loss(a, y):
    return a-y

def s(x):
    return 1.0/(1+np.exp(x))

def ds(x):
    return s(x)*(1-s(x))

def ReLU(x):
    return np.where(x>0, x,0.1*x)

def d_ReLU(x):
    return np.where(x>0, 1, 0.1)

def hot(y,n):
    h = np.zeros(n)
    h[y]=1
    return h

def d(fct):
    ddx = {s:ds, loss:d_loss, ReLU:d_ReLU}
    return ddx[fct]

# Activation function
act = ReLU


class neural_network:
    '''
        Feedforward inspired by M. Nielsen's
        'Neural Networks and Deep Learning' book
    '''
    
    def __init__(self, layers):
        '''
            params:
                layers : liste of layers dimensions
                         ex. neural_network([20,20,5])
        '''
        
        self.layers = layers
        self.L = len(layers)
        
        #weights/biases initialisation
        self.W_list = [1/np.sqrt(784)*np.random.randn(self.layers[k+1], layers[k]) for k in range(self.L-1)]
        self.b_list = [np.random.randn(l,1) for l in layers[1:]]
        
    def forward(self,x):
        for W,b in zip(self.W_list, self.b_list):
            x = act(W @ x + b)
        return x
    
    def backward(self, x, y):
        #forward pass
        z_list  = []
        a_list = [x]
        for W,b in zip(self.W_list, self.b_list):
            z = W @ x + b
            z_list.append(z)
            x = act(z)
            a_list.append(x)
        
        #delta_list = [] # δ's
        
        delta= d(loss)(a_list[-1],y)* d(act)(z_list[-1]) # delta L
    
        grad_W_list = []
        grad_b_list = []
        
        # Backprop starts
        grad_W = delta @ a_list[-2].T
        grad_b = delta
        
        grad_W_list.insert(0,grad_W)
        grad_b_list.insert(0,grad_b)
        
        for l in range(self.L-2,0,-1): #l : L-2 -> 1
            # Nouveau delta
            delta = (self.W_list[l].T @ delta) * d(act)(z_list[l-1])
            
            # grads
            grad_W = delta @ a_list[l-1].T
            grad_b = delta
        
            grad_W_list.insert(0,grad_W)
            grad_b_list.insert(0,grad_b)
        
        return grad_W_list, grad_b_list
    
    def train(self, X, Y, lr=1, mini_batch_size=10, nb_epoch=1):
        n = len(X[0])
        nb_mini_batch = n//mini_batch_size
        acc = []
        
        for k1 in range(nb_epoch):
            for k2 in range(n):
                grad_W, grad_b = self.backward(X[:,k2].reshape((len(X[:,k2]),1)), Y[:,k2].reshape((len(Y[:,k2]),1)))
                
                for l in range(len(self.W_list)):
                    self.W_list[l] -= lr*grad_W[l]
                    self.b_list[l] -= lr*grad_b[l]
            acc.append(self.test(X_test, Y_test))
        return acc
    
    def predict(self,x):
        a = self.forward(x)
        return np.argmin(np.array([loss(a, hot(i,len(a)).reshape((len(a),1))) for i in range(len(a))]))
    
    def test(self, X, Y):
        succes = 0
        for k in range(len(X[0])):
            y_pred = self.predict(X[:,k].reshape((len(X[:,k]),1)))
            if y_pred == np.argmax(Y[:,k]):
                succes+=1
        return succes/len(X[0])

