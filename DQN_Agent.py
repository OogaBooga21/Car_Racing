import torch
from torch import nn
import numpy as np
import random
import time
from DQN_Replay_Mem import Replay_Memory
from DQN import Network
from torch.utils.tensorboard import SummaryWriter


class RL_Agent:
    def __init__(self, env,memory_size ,epsilon, epsilon_end, epsilon_decay, batchsize, gamma, target_update_freq):
        
        self.env = env 
        stack_frames, self.img_s_h, self.img_s_w = self.env.env_state_shape()
        
        
        output_layer_size = self.env.action_space_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.online_network = Network(stack_frames, self.img_s_h, self.img_s_w, output_layer_size).to(self.device)
        self.online_network.train()
        
        self.target_network = Network(stack_frames, self.img_s_h, self.img_s_w, output_layer_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval() # make a copy and use it only for eval  
        
        self.best = Network(stack_frames, self.img_s_h, self.img_s_w, output_layer_size).to(self.device)
        self.best.load_state_dict(self.online_network.state_dict())
        
        self.target_update_freq = target_update_freq
        
        self.online_network.to(self.device)
        self.target_network.to(self.device)
        print("Running on ", self.device, ": ")
        print(self.online_network) 
        #for the neural networks
        
        self.mem_cap = memory_size
        self.replay_memory = Replay_Memory(memory_size, stack_frames, self.img_s_h, self.img_s_w)
        #replay memory
        
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        #epsilon stuff
        
        self.gamma = gamma
        self.batch_size = batchsize
        self.episode_count = 0
        #miscelanious parameters
        
        self.highscore = -9999
        self.reward_cutof = -3.0
        #score stuff

        self.writer = SummaryWriter('logs/gradient_logging') #Logging
        self.fill_memory()
        
        
    def save_model_state(self, network, filename):
        torch.save(network.state_dict(), filename)

    def load_model_state(self, filename,test):
        if torch.cuda.is_available():  # Check for GPU availability
            map_location = torch.device('cuda')  # Map to GPU if possible
        else:
            map_location = torch.device('cpu')  # Fallback to CPU

        self.online_network.load_state_dict(torch.load(filename, map_location=map_location))
        if test:
            self.online_network.eval()  # Set to evaluation mode 
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - (1 - self.epsilon_end)/self.epsilon_decay, self.epsilon_end)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict()) #update network
    
    def fill_memory(self):
        state_count = 0
        while state_count < self.mem_cap:
            state = self.env.reset()
            terminal = False
            while not terminal:
                action = self.pick_action(state)
                next_state,reward, terminal = self.env.step(action)
                
                if reward > self.reward_cutof:
                    self.replay_memory.store_memory(state,  # Pick, see consequences, store 
                                                    action,
                                                    reward,
                                                    terminal,
                                                    next_state)
                
                state=next_state
                state_count+=1
                
                if state_count>=self.mem_cap:
                     terminal = True
                
                percent = round(100 * state_count /  self.mem_cap, 2)
                filled_length = int(50 * state_count //  self.mem_cap)
                bar = f'[{filled_length * "#"}{"-" * (50 - filled_length)}]'
                print(f'{"Filling Replay Memory: "} {bar} {percent:.2f}% {" Done."}', end="\r")
                if state_count ==  self.mem_cap:
                    print()

    def pick_action(self,state):
        if random.random() < self.epsilon:
            return self.env.random_action()
        else:        
            with torch.no_grad():
                state = torch.tensor(np.stack(state, axis=0), device=self.device, dtype=torch.float32)
                action = self.online_network.forward(state.unsqueeze(0)) #might get rid of some code in forward
            return torch.argmax(action).item() # "BEST" ACTION

    def learn(self):
        if self.replay_memory.fill_rate() < self.batch_size:
            print("Replay memory has less samples than the batch size, fill memory before training ....")
            return # This should literally never happen, but ok

        states, actions, rewards, terminals, next_states = self.replay_memory.random_memory_batch(self.batch_size) # Get a batch

        predicted_q_values = self.online_network.forward(states)

        target_q_values = self.target_network.forward(next_states).detach() #It's a copy, we don't need to punish twice or old mistakes
        
        target_q_values = torch.max(target_q_values, dim=1, keepdim=True).values #Which action has higher value

        rewards = rewards.unsqueeze(1) # Preparing data for Bellman's
        terminals = terminals.unsqueeze(1)  # Same thing yurr, but with conversion cuz
        actions = actions.unsqueeze(1) # Y'alredy no wasgoinon
        
        target_q_values = (rewards + self.gamma * target_q_values) * (1 - terminals)
        predicted_q_values = predicted_q_values.gather(1, actions)
        
        self.online_network.optimizer.zero_grad()
        
        loss = nn.functional.mse_loss(predicted_q_values, target_q_values)#.mean()
        loss.backward()
        self.online_network.optimizer.step()
        self.online_network.scheduler.step()    
                
        return loss
    
    def train(self,train_episodes):
        print('Training...')      
        reward_history = []
        loss_history = []
        step_history = []

        for episode_count in range(train_episodes): # Do it for n episodes
            self.episode_count+=1
            step_count = 0
            state = self.env.reset() #reset env, get first state, and process it (for the first pick_action)
            terminal=False
            episode_reward=0

            while not terminal: # While not the end of the episode
                action = self.pick_action(state)
                next_state, reward, terminal = self.env.step(action)
                
                if reward > self.reward_cutof:
                    self.replay_memory.store_memory(state,  # Pick, see consequences, store 
                                                    action,
                                                    reward,
                                                    terminal,
                                                    next_state)

                if step_count % self.target_update_freq == 0:
                    self.update_target_network() # Update Target ?

                state=next_state
                episode_reward += reward
                step_count += 1
        
            if(episode_count % 2 == 0):
                for _ in range(2):
                    loss_copy = self.learn() # Call the smart function
                    loss_history.append(loss_copy.item())
                    
            self.update_epsilon()
            
            reward_history.append(episode_reward)
            step_history.append(step_count)
            current_avg_score = np.mean(reward_history[-10:]) 
            current_avg_loss = np.mean(loss_history[-10:])
            current_avg_step = np.mean(step_history[-10:])

            if(episode_count % 10 == 0): # Update user
                print('ep:{}, HS: {}, BA:{}, LA:{},SA:{},E:{}'.format(episode_count, int(self.highscore), int(current_avg_score), current_avg_loss, int(current_avg_step) ,self.epsilon))                   
                print([int(reward) for reward in reward_history[-10:]], time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec)
                # Inside RL_Agent.train() method, after updating reward_history, loss_history, etc.
                self.writer.add_scalar('Training/Average Reward', current_avg_score, episode_count)
                self.writer.add_scalar('Training/Average Loss', current_avg_loss, episode_count)
                self.writer.add_scalar('Training/Step Count', current_avg_step, episode_count)
                self.writer.add_scalar('Training/Epsilon', self.epsilon, episode_count)
                
                self.save_model_state(self.best,"best.pt")
        
            if episode_reward >= self.highscore: # Save ?
                self.highscore = episode_reward
                self.best.load_state_dict(self.online_network.state_dict()) #save for later
            
        self.save_model_state(self.best,"best.pt") #actual save
        
    def test(self, test_episodes):
        
        for episode_count in range(test_episodes):
            state=self.env.reset()
            done=False
            episode_reward = 0
            
            while not done: #should be fixed? 
                action = self.pick_action(state)
                new_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = new_state

            print('ep:{}, ep score: {}'.format(episode_count,episode_reward))