'''
    Script : calcul de la solution exact pour la dynamique d'une poutre
    encastree.

'''

import math

def f(x):
    return x**2
    

Beta=[1.87510406871196, 4.69409113297418, 7.85475743823761, 10.9955407348755, 14.1371683910465, 17.2787595320882, 20.4203522510413, 23.5619449018064, 26.7035375555183, 29.8451302091028, 32.9867228626928, 36.1283155162826, 39.2699081698724, 42.4115008234622, 45.5530934770520]

def sinh(x):
    return math.sinh(x) 
def cosh(x):
    return math.cosh(x)
def sin(x):
    return math.sin(x)
def cos(x):
    return math.cos(x)

def lam(n,L):
    return Beta[n-1]/L

def X(n,x):
    return (sinh(lam(n,L)*L)+sin(lam(n,L)*L))*(cosh(lam(n,L)*x)-cos(lam(n,L)*x))-(cosh(lam(n,L)*L)+cos(lam(n,L)*L))*(sinh(lam(n,L)*x)-sin(lam(n,L)*x))

def integral_de(g,a,b,N):
    dx= (b-a)/N
    somme =0
    
    for i in range (1,N):
        
        somme += 2*g(a+i*dx)
        
    
    return dx/2*(somme+g(a)+g(b))

def D(g,x): #Differences finies d'ordre 2
    return (g(x+0.000000001)-g(x-0.000000001))/0.000000002

def D2(g,x): #Differences finies d'ordre 2
    return (g(x+0.000000001)- 2*g(x) +g(x-0.000000001))/(0.000000001**2)

def norme_X(n):
    
    def X2(x):
        return X(n,x)**2
    
    return integral_de(X2,0,L,100)

def F4(f,n):
    
    def fX(x):
        return f(x)*X(n,x)
    
    return integral_de(fX,0,L,Nb)    

def iF4(F,t,x,n_max):
    serie = 0
    for n in range (1,n_max+1):
        serie += ((norme_X(n))**-1) *F(t,n)*X(n,x)
    
    return serie

    #n:mode, L:longueur, x:distance
    #fct X represente profil du mode n, amplitude en fct x
#--------------------------------------------------------------------------
# ------- Model Poutre Cantilever--------

Nb= 25 #precisions des integrales
n_de_mode = 10


# Configuration
L = 1.0
E = 1.0 #Module de Young
I = 1.0 #Moment d'aire
rho = 1.0 # Densite
A = 0.0026 # Aire de section

def h(t): #Hauteur en fonction du temps
    return 0.0

def q(x,t): #Charge/unitee de longeur
    return -5*t

def po(x): #Profil initial
    return 0.0
#--------------------------------------

alp = math.sqrt(rho*A/(E*I))

def l(n):
    return lam(n,L)

def w(n): #Frequence Ang
    return (Beta[n-1])**2 * math.sqrt(E*I/(rho*A))*(L**(-2))

def g1(x):
    return -D(h,0)

def f1(x):
    return po(x)-h(0)
    
def Q1(t,n):

    def integrant(x):
        return q(x,t)-(alp**2)*D2(h,t)
    return F4(integrant,n)

def V(t,n):

    an = F4(f1,n)
    bn = F4(g1,n)
    def integrante(s):
        return Q1(s,n)*sin((l(n)**2)/alp *(t-s))
    I = integral_de(integrante, 0,t,Nb)

    return an*cos((l(n)**2)/alp *t)+ bn*sin((l(n)**2)/alp *t)+ alp/(l(n)**2) * I

def u(x,t):

    return h(t)+iF4(V,t,x,n_de_mode)

def genererSol(t_max, step_x,step_t):
    sol = []
    ti = 0
    
    while round(ti,4) <= t_max:
        xi = 0
        solx = []
        while round(xi,4) <= L:
            solx.append([round(xi,4),round(u(xi,ti),10)])
            xi += step_x
            
        sol.append(solx)
        ti +=step_t
    return sol

print(genererSol(0.3,0.1,0.1))
    
