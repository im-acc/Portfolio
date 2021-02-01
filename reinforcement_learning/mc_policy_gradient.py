import gym
import tensorflow as tf
import numpy as np
from numpy import array as ar
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')


class pg_agent:
    """
        Monte Carlo Policy Gradient without baseline : 2 hidden layer, softmax neural net
    """
    
    
    def __init__(self, n_states_dim, n_action):
        self.n_states_dim = n_states_dim
        self.n_action = n_action
        self.n_action_list = np.arange(n_action)
        
        self.memory = np.zeros((0,n_states_dim*2+2)) # episode_memory
        #self.returns = [] # list of discounted returns for each time step
        
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'), 
            tf.keras.layers.Dense(n_action, activation='softmax')
        ])
        
        def pg_loss(y_true, y_pred):
            return -K.sum(K.log(K.clip(y_pred,1e-8, 1-1e-8)) * y_true)
        
        self.model.compile(optimizer='adam', loss=pg_loss)
        
        self.loss = []
        self.steps = []
    
    # Selects action   
    def predict(self, state):
        probs = self.model(np.array([state]))
        return np.random.choice(self.n_action_list, p=ar(probs[0]) )

    
    def post_episode_learning(self, mini_batch_size=10, Y=0.95):
        ''' TODO : discout reward Y '''
        batch = self.memory
            
        states = batch[:, :self.n_states_dim]
        actions = batch[:, self.n_states_dim]
        new_states = batch[:, self.n_states_dim+1:-1]
        rewards = batch[:, -1]
        
        returns = np.cumsum(rewards[::-1])[::-1]
        returns -= returns.mean()
        returns /= returns.std()

        history = self.model.fit(states, returns, batch_size=mini_batch_size, epochs=1, verbose=0)
        self.loss = self.loss + history.history['loss']

    
    def train(self, env, n_episode=100, mini_batch_size=30, Y=0.95):
        '''
            Training loop : trains the network on the provided OpenAI gym environement.
        '''
        for n in range(n_episode):
            obs = env.reset()
            
            for t in range(200):
                action = self.predict(obs)
                new_obs, R, done, info = env.step(action)
                if done:
                    R -= 15
                    self.steps.append(t)
                self.memory = np.vstack((self.memory, np.concatenate( ( obs, ar([action]), new_obs, ar([R]) ) )))
                
                if done:
                    break
                
                obs = new_obs
            
            self.post_episode_learning(mini_batch_size=mini_batch_size, Y=0.95)
            
            self.memory = np.zeros((0, self.n_states_dim*2+2)) # clear episode_memory
                
        env.close()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = pg_agent(4,2)
    agent.train(env, n_episode=200)
