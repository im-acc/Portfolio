import gym
import numpy as np
from numpy import array as ar
from matplotlib import pyplot as plt
import tensorflow as tf

class q_net:
    """
        Approximate neural network based Q learning for continuous states 
        space and discrete action space. e-greedy exploration. Fully connected
        network, 2 relu hidden layers.
    """
    
    
    def __init__(self, n_states_dim, n_action):
        self.n_states_dim = n_states_dim
        self.n_action = n_action
        
        self.memory_queue = np.zeros((0,n_states_dim*2+2)) # experience replay stack
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'), 
            tf.keras.layers.Dense(n_action)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        self.loss = []
        self.steps = []
    
    # Selects best action   
    def predict(self, state):
        return np.argmax(self.model(np.array([state])))

    
    def learn_from_memory(self, mini_batch_size=30, Y=0.95):
        '''
            Experience replay : trains for 1 epoch on the entire
            memory queue. Uses R + Y max_a Q(s',a) as target.
        '''
        
        batch = self.memory_queue
            
        states = batch[:, :self.n_states_dim]
        actions = batch[:, self.n_states_dim]
        new_states = batch[:, self.n_states_dim+1:-1]
        rewards = batch[:, -1]
            
        target = self.model.predict(states)

        q_target = rewards + Y * np.max(self.model(new_states), axis=1)
        for i in range(len(batch)):
                target[i,int(actions[i])] = q_target[i]
                
        history = self.model.fit(states, target, batch_size=mini_batch_size, epochs=1, verbose=0)
        self.loss = self.loss + history.history['loss']

    
    def train(self, env, n_episode=100, mini_batch_size=30, eps=0.20, Y=0.95, memory_queue_length=10_000):
        '''
            Training loop : trains the network on the provided OpenAI gym environement.
        '''
        for n in range(n_episode):
            obs = env.reset()
            
            for t in range(200):
                
                if np.random.rand() < eps:
                    action = np.random.randint(self.n_action)
                else:
                    action = self.predict(obs)
                
                new_obs, R, done, info = env.step(action)
                if done:
                    R -= 15
                    self.steps.append(t)
                self.memory_queue = np.vstack((self.memory_queue, np.concatenate( ( obs, ar([action]), new_obs, ar([R]) ) )))
                
                if done:
                    break
                
                obs = new_obs
            
            self.learn_from_memory(mini_batch_size=mini_batch_size, Y=0.95)
            if len(self.memory_queue) > memory_queue_length:
                self.memory_queue = self.memory_queue[:-memory_queue_length]
                
        env.close()

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = q_net(4,2)
    
    agent.train(env, n_episode=100, eps=0.50)
    agent.train(env, n_episode=200, eps=0.40)
    agent.train(env, n_episode=300, eps=0.30)
    agent.train(env, n_episode=200, eps=0.20)
    agent.train(env, n_episode=500, eps=0.10)
    plt.plot(agent.steps)
    plt.plot(agent.loss)
