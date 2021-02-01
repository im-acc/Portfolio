import numpy as np
import copy
from matplotlib import pyplot as plt

class TabuSearchSolver:
    
    def __init__(self, adjMat):
        self.adjMat = adjMat
        self.n = len(self.adjMat)
        self.tabu_list = dict()
        self.nn = nearestNearbourHeuristic(adjMat)
        self.sol_min = None
        self.historic = []
    
    def solve(self, niter=1000, nb_random_sample=30, list_length=30):
        # Initial solution
        sol = self.nn.solve()
        self.sol_min = copy.deepcopy(sol)
        self.historic = [sol.cost]
        oldest_move = None
        
        # Iteration
        for i in range(niter):
            
            # Local Search step
            move = self.tabu_step(sol, nb_random_sample)

            # Update historic
            #print(move)
            #print(self.tabu_list)
            #print(sol)
            self.historic.append(sol.cost)

            if sol.cost < self.sol_min.cost:
                self.sol_min = copy.deepcopy(sol)
            
            # Append & update tabu list
            self.tabu_list[move] = i
            for k in self.tabu_list:
                if self.tabu_list[k]==i-list_length:
                    self.tabu_list.pop(k, 'Not found')
                    break
        
        return self.sol_min

    def tabu_step(self, sol, nb_random_sample=30):
        min_i = None 
        min_j = None
        min_cost = 10**10
        
        idx = 0
        while idx < nb_random_sample:
            nombres = np.random.randint(0,self.n,2)
            i = np.min(nombres)
            j = np.max(nombres)
            if i!=j and not (i==0 and j==self.n-1): #(np.absolute(j-i)<self.n-1): #
                '''if i==0 and j==self.n-1:
                    temp = j
                    j = i
                    i = temp
                '''
                new_cost, ville_i, ville_j = self.tabu_switch_2opt(sol, i, j)

                if ( (i,j) not in self.tabu_list or new_cost < self.sol_min.cost):
                    if (new_cost < min_cost):
                        min_i = i
                        min_j = j
                        min_cost = new_cost    
                    idx += 1
        #print(min_i, min_j)
        sol.swap_2opt(min_i, min_j, min_cost)
        #print(sol)
        return (min_i, min_j)

    def tabu_switch(self, sol, i, j):
        if np.absolute(i-j)==0:
            return (sol.cost, sol.path[i], sol.path[j])
        
        # find the index of previous and next
        next_i = self.next_idx(i)
        prev_i = self.prev_idx(i)
                
        next_j = self.next_idx(j)
        prev_j = self.prev_idx(j)

        indice1_prev = sol.path[prev_i]
        indice1 = sol.path[i]
        indice1_next = sol.path[next_i]
        indice2_prev = sol.path[prev_j]
        indice2 = sol.path[j]
        indice2_next = sol.path[next_j]


        cost = sol.cost
        
        if np.absolute(i-j)==1 or np.absolute(i-j) == self.n-1: # checker avec debut et loop
            cost -= self.adjMat[indice1_prev][indice1]
            cost -= self.adjMat[indice2][indice2_next]
            cost -= self.adjMat[indice1][indice2]
            
            cost += self.adjMat[indice1_prev][indice2]
            cost += self.adjMat[indice2][indice1]
            cost += self.adjMat[indice1][indice2_next]
            return (cost, indice1, indice2)
        
        # remove the cost
        cost -= self.adjMat[indice1_prev][indice1]
        cost -= self.adjMat[indice1][indice1_next]

        cost -= self.adjMat[indice2_prev][indice2]
        cost -= self.adjMat[indice2][indice2_next]

        # add the cost
        cost += self.adjMat[indice1_prev][indice2]
        cost += self.adjMat[indice2][indice1_next]

        cost += self.adjMat[indice2_prev][indice1]
        cost += self.adjMat[indice1][indice2_next]
        return (cost, indice1, indice2)

    # 2-opts new cost switch avec TSP symetrique <----
    def tabu_switch_2opt(self, sol, i, j):        
        # find the index of previous and next
        next_i = self.next_idx(i)
        prev_i = self.prev_idx(i)
                
        next_j = self.next_idx(j)
        prev_j = self.prev_idx(j)

        indice1_prev = sol.path[prev_i]
        indice1 = sol.path[i]
        indice1_next = sol.path[next_i]
        indice2_prev = sol.path[prev_j]
        indice2 = sol.path[j]
        indice2_next = sol.path[next_j]
        
        cost = sol.cost
        cost -= self.adjMat[indice1_prev][indice1]
        cost -= self.adjMat[indice2][indice2_next]
        
        cost += self.adjMat[indice1_prev][indice2]
        cost += self.adjMat[indice1][indice2_next]
        
        return (cost, indice1, indice2)

    def next_idx(self, i):
        if (i + 1) == self.n:
            return 0
        else:
            return i+1
    
    def prev_idx(self, i):
        if i==0:
            return self.n-1
        else:
            return i-1        


class nearestNearbourHeuristic:

    def __init__(self, adjMat):
        self.adjMat = adjMat
        self.n = len(self.adjMat)

    def solve_d(self, depart):
        cost = 0
        # TODO
        noeud = depart #np.random.randint(0,self.n) #depart
        #depart = noeud
        path = [noeud]
        visite = np.zeros(self.n) # 1 si deja visite, 0 sinon
        for _ in range(self.n-1): # itere pour tracer le chemin
            visite[noeud] = 1
            idx_min = 0 # index
            while visite[idx_min] != 0: # On trouve le premier candidat
                idx_min += 1
            for noeud_next in range(self.n): # on itere pour trouver l'arete minimum
                if not visite[noeud_next]:
                    if self.adjMat[noeud][noeud_next] < self.adjMat[noeud][idx_min]:
                        idx_min = noeud_next
            path.append(idx_min)
                        
            # on met a jour le cout de la solution
            cost += self.adjMat[noeud][idx_min]
            noeud = idx_min
        # on ajoute le cout de retour
        cost += self.adjMat[noeud][depart]

        return Solution(path, cost)
    
    def solve(self):
        cost=10**10
        sol_min = None
        for i in range(self.n):
            sol = self.solve_d(i)
            if sol.cost < cost:
                sol_min = sol
                cost = sol_min.cost
        return sol_min
    
class Solution:
    
    def __init__(self, path=[], cost=0):
        self.path = path
        self.cost = cost

    def swap(self, i, j, cost):
        self.cost = cost
        temp = self.path[i]
        self.path[i] = self.path[j]
        self.path[j] = temp
    
    def swap_2opt(self, i, j, cost):
        self.cost=cost
        self.path = self.path[:i] + self.path[i:j+1][::-1] + self.path[j+1:]
    
    def __str__(self):
        return "path : " + str(self.path) + ", cost : "  + str(self.cost)
    
