import numpy

class cartpole_tabular_Q():
    '''
        Simple tabular Q learning for cartpole problem with
        discrete action space.
    '''
    
    
    def __init__(self, n=5):
        self.Q = np.zeros((n,n,n,n,2))
        self.Action = [0,1]
        self.n = n
        self.loss = []
    
    def predict(self, dis_obs):
        return np.argmax(self.Q[tuple(dis_obs)])
    
    def discret_obs(self, obs):
        normalized_obs = np.array([obs[0]/(2*4.8), softsig(obs[1]), obs[2]/(2*0.418), softsig(obs[3])])
        
        seps = np.linspace(-1,1,self.n+1)
        dis_obs = np.argmax([ob < seps for ob in normalized_obs], axis=1) - 1
        return dis_obs
    
    def iteration(self,nepisode=100, eps=0.15, alpha=0.01):

        for i in range(nepisode):
            obs = env.reset()
            for t in range(200):
                dis_obs = self.discret_obs(obs)
                
                if np.random.rand() < eps:
                    action = np.random.randint(0,2)
                else:
                    action = self.predict(dis_obs)

                obs, R, done, info = env.step(action)

                new_dis_obs = self.discret_obs(obs)
                q = self.Q[tuple(dis_obs)]
                q[action] += alpha * (R +  np.max(self.Q[tuple(new_dis_obs)]) - q[action])

                if done:
                    break
            self.loss.append(t)
        env.close()            
    
    def __call__(self,obs):
        return self.predict(obs)
