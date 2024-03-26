import torch
from torch import nn
import numpy as np
import random

class RL_Agent:
    def __init__(self, env, target_network, online_network, replay_memory, epsilon, epsilon_end, epsilon_decay, batchsize, gamma, target_update_freq):
        self.env = env
        self.target_network = target_network
        self.online_network = online_network
        self.replay_memory = replay_memory
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batchsize
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.mem_cap = self.replay_memory.get_capacity()
        self.data_shape = int(np.prod(self.env.observation_space.shape))
        self.highscore = -np.inf
        
        self.update_target_network()
        self.target_network.eval() # make a copy and use it only for eval  
        
    def save_model_state(self, network, filename):
        torch.save(network.state_dict(), filename)

    def load_mode_state(self, filename):
        self.online_network.load_state_dict(torch.load(filename))
        #remember to make it .eval when testing
        
    def clean_states(self,state_batch):
        clean_states_array = []
        for state in state_batch:
            # Check if the state is already a NumPy array
            if isinstance(state, np.ndarray):
                clean_state = state.flatten().astype(float)
            else:
                # Extract the first element from the tuple or list
                clean_state = state[0].flatten().astype(float)
            clean_states_array.append(clean_state)

        matrix_result = np.array(clean_states_array)
        return matrix_result
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - (1 - self.epsilon_end)/self.epsilon_decay, self.epsilon_end)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict()) #update network
        # return self.target_network

    def fill_memory(self):
        print("Filling Replay Memory")
        for _ in range(self.mem_cap):
            state = self.env.reset()
            terminal = False
            while not terminal:
                action = self.pick_action(state)
                next_state, reward , terminal, _, _ = self.env.step(action)
                
                self.replay_memory.store_memory(state, action, reward, terminal, next_state)
                state=next_state
            terminal = False

    def pick_action(self,state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample() # RANDOM ACTION
        else:        
            with torch.no_grad():
                if len(state) != int(np.prod(self.env.observation_space.shape)):
                    action = self.online_network.forward(state[0])
                else:
                    action = self.online_network.forward(state)
            return torch.argmax(action).item() # INDEX OF "BEST" ACTION

    def learn(self):
        if self.replay_memory.fill_rate() < self.batch_size:
            print("Replay memory has less samples than the batch size, skipping this learning round ....")
            return # This should literally never happen, but ok

        states, actions, rewards, terminals, next_states = self.replay_memory.random_memory_batch(self.batch_size) # Get a batch

        clean_states = self.clean_states(states) #Clean and forward the states (clean cuz sometimes it's a touple)
        clean_next_states = self.clean_states(next_states)  
        predicted_q_values = self.online_network.forward(clean_states)
        target_q_values = self.target_network.forward(clean_next_states)
        
        target_q_values = torch.max(target_q_values, dim=1, keepdim=True).values #Which action has higher value

        rewards = torch.from_numpy(rewards).float() # Preparing data for Bellman's
        rewards = rewards.unsqueeze(1)
        

        terminals = torch.from_numpy(terminals).float()  # Same thing yurr, but with conversion cuz
        terminals = terminals.unsqueeze(1)
        
        actions = torch.from_numpy(actions) # Y'alredy no wasgoinon
        actions = actions.unsqueeze(1)
        
        target_q_values = (rewards + self.gamma * target_q_values) * (1 - terminals)
        predicted_q_values = predicted_q_values.gather(1, actions)
        
        self.online_network. optimizer.zero_grad()
        loss = nn.functional.mse_loss(predicted_q_values, target_q_values).mean()
        loss.backward()
        self.online_network.optimizer.step()        

    def train(self,train_episodes):      
        
        step_count = 0
        reward_history = []
        best = self.online_network

        for episode_count in range(train_episodes): # Do it for n episodes
            state = self.env.reset()
            terminal=False
            episode_reward=0

            while not terminal: # While not the end of the episode
                action = self.pick_action(state)
                new_state, reward, terminal,_,_ = self.env.step(action)
                self.replay_memory.store_memory(state,action,reward,terminal,new_state) # Pick, see consequences, store 

                self.learn() # Call the smart function

                if step_count % self.target_update_freq == 0:
                    self.update_target_network() # Update Target ?

                state = new_state
                episode_reward += reward
                step_count += 1

            self.update_epsilon()
            reward_history.append(episode_reward)

            current_avg_score = np.mean(reward_history[-50:])

            if(episode_count % 50 == 0): # Update user
                print('ep:{}, ep_Score: {}, batch_avg:{}, updated_epsilon:{}'.format(episode_count, episode_reward, current_avg_score, self.epsilon))                   
                            
        
            if current_avg_score >= self.highscore: # Save ? 
                self.highscore = current_avg_score
                best = self.online_network #save for later

        self.save_model_state(best,"huh.pt") #actual save
        
    def test(self, env, test_episodes):
        
        for episode_count in range(test_episodes):
            state=env.reset()
            done=False
            episode_reward = 0

            while not done:
                action = self.pick_action(state)
                new_state, reward, done, _, _ = env.step(action)
                state = new_state
                episode_reward += reward

            print('ep:{}, ep score: {}'.format(episode_count,episode_reward))
        env.close()
