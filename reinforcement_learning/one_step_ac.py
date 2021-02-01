import gym
import tensorflow as tf
import numpy as np
from numpy import array as ar
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K

tf.keras.backend.set_floatx('float64')

class ac_agent:
    """
       Approxiomate one step TD Actor-critic nn based reinforcement learning control :  
    """
    
    def __init__(self, n_states_dim, n_action):
        self.n_states_dim = n_states_dim
        self.n_action = n_action
        self.n_action_list = np.arange(n_action)
        
        self.memory = np.zeros((0,n_states_dim*2+2)) # episode_memory
        #self.returns = [] # list of discounted returns for each time step
        
        self.policy_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'), 
            tf.keras.layers.Dense(n_action, activation='softmax')
        ])
        
        self.value_model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'), 
            tf.keras.layers.Dense(n_action)
        ])
        self.value_model.compile(optimizer='adam', loss='mse')
        
        def pg_loss(y_true, y_pred):
            return -K.sum(K.log(K.clip(y_pred,1e-8, 1-1e-8)) * y_true)
        
        self.policy_model.compile(optimizer='adam', loss=pg_loss)
        
        self.loss = []
        self.steps = []
    
    # Selects action   
    def predict(self, state):
        probs = self.policy_model(np.array([state]))
        return np.random.choice(self.n_action_list, p=ar(probs[0]) )
    
    def learn(self, s, a, ss, r, done, Y):
        s, ss = (ar([s]), ar([ss]))
        vs = self.value_model(s)
        vss = self.value_model(ss)
        
        TD_error = r - vs
        target = r
        if not done:
            target += Y*vss
            TD_error += Y*vss
        
        history_policy = self.value_model.fit(s, target, batch_size=1, epochs=1, vervose=0)
        history_value = self.policy_model.fit(s, TD_error, batch_size=1, epochs=1, vervose=0)
        self.loss = self.loss + history.history_policy['loss']
        
        
    def train(self, env, n_episode=100, mini_batch_size=30, Y=1):
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
                self.learn(obs, action, R, new_obs, done, Y)
                if done:
                    break
                obs = new_obs

        env.close()

'''
env = gym.make('CartPole-v1')
agent = pg_agent(4,2)
agent.train(env, n_episode=200)
plt.plot(agent.steps)
'''
