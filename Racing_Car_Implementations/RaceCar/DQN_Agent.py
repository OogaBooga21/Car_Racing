import torch
from torch import nn
import numpy as np
import random
import time
import cv2
from collections import deque
from DQN_Replay_Mem import Replay_Memory
from DQN import Network


class RL_Agent:
    def __init__(self, env, memory_size ,epsilon, epsilon_end, epsilon_decay, batchsize, gamma, target_update_freq,rescale_factor ,stopping_reward, stopping_time, stopping_steps, initial_skip_frames, skip_frames, stack_frames):
        
        self.env = env
        dummy_state, _ = env.reset()
        img_height = dummy_state.shape[0]
        img_width = dummy_state.shape[1]
        #images
        
        self.resize = rescale_factor
        self.img_s_h = int(rescale_factor * img_height)
        self.img_s_w = int(rescale_factor * img_width)
        #memory images
        
        output_layer_size = int(env.action_space.n)
        # self.online_network = Network(stack_frames, img_height, img_width, output_layer_size)
        # self.target_network = Network(stack_frames, img_height, img_width, output_layer_size) 
        self.online_network = Network(stack_frames, self.img_s_h, self.img_s_w, output_layer_size)
        self.target_network = Network(stack_frames, self.img_s_h, self.img_s_w, output_layer_size) 
        self.target_update_freq = target_update_freq
        self.update_target_network()
        self.target_network.eval() # make a copy and use it only for eval  
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
        #miscelanious parameters
        
        self.highscore = -np.inf
        self.stopping_reward= stopping_reward
        self.stopping_time = stopping_time
        self.stopping_steps = stopping_steps
        #stopping and scoring
        
        self.initial_skip_frames = initial_skip_frames
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        #temporal stuff ?
        
        self.fill_memory()
        
        
    def save_model_state(self, network, filename):
        torch.save(network.state_dict(), filename)

    def load_mode_state(self, filename):
        self.online_network.load_state_dict(torch.load(filename))
        #remember to make it .eval when testing
        
    def preprocess_state(self,state):
        gray_image = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) / 255.0
        resized_img = cv2.resize(gray_image, (self.img_s_h, self.img_s_w), interpolation=cv2.INTER_AREA)
        return resized_img#.flatten()
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - (1 - self.epsilon_end)/self.epsilon_decay, self.epsilon_end)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict()) #update network

    def env_reset(self):
        s, _ = self.env.reset()
        
        for _ in range (self.initial_skip_frames):
            s,_,_,_,_ = self.env.step(0)
        
        s=self.preprocess_state(s)
        
        for _ in range(self.stack_frames):
            self.frame_stack.append(s)
            
    def env_step(self, action):
        reward = 0
        for _ in range(self.skip_frames):
            s,r,terminal, _, _ = self.env.step(action)
            reward+=r
            s = self.preprocess_state(s)
            
            if terminal:
                break
            
            # s = self.preprocess_state(s)
        self.frame_stack.append(s)       
        return reward, terminal

    def fill_memory(self):
        print("Filling Replay Memory...")
        state_count = 0
        while state_count < self.mem_cap:
            self.env_reset()
            terminal = False
            episode_reward = 0
            while not terminal:
                action = self.pick_action()
                initial_state = self.frame_stack
                reward, terminal = self.env_step(action)
                self.replay_memory.store_memory(initial_state,  # Pick, see consequences, store 
                                                action,
                                                reward,
                                                terminal,
                                                self.frame_stack)
                
                episode_reward+=reward
                state_count+=1
                
                if episode_reward <= self.stopping_reward or state_count>self.mem_cap:
                    terminal = True
                print(state_count, self.mem_cap)                          

    def pick_action(self):
        if random.random() < self.epsilon:
            return self.env.action_space.sample() # RANDOM ACTION/S
        else:        
            with torch.no_grad():
                state = np.stack(self.frame_stack)
                action = self.online_network.forward(state)
            return torch.argmax(action).item() # INDEX OF "BEST" ACTION

    def learn(self):
        if self.replay_memory.fill_rate() < self.batch_size:
            print("Replay memory has less samples than the batch size, fill memory before training ....")
            return # This should literally never happen, but ok

        states, actions, rewards, terminals, next_states = self.replay_memory.random_memory_batch(self.batch_size) # Get a batch
        
        predicted_q_values = self.online_network.forward(states)
        target_q_values = self.target_network.forward(next_states).detach() #It's a copy, we don't need to punish twice or old mistakes
        
        target_q_values = torch.max(target_q_values, dim=1, keepdim=True).values #Which action has higher value

        rewards = torch.from_numpy(rewards).float() # Preparing data for Bellman's
        rewards = rewards.unsqueeze(1)
        

        terminals = torch.from_numpy(terminals).float()  # Same thing yurr, but with conversion cuz
        terminals = terminals.unsqueeze(1)
        
        actions = torch.from_numpy(actions) # Y'alredy no wasgoinon
        actions = actions.unsqueeze(1)
        
        target_q_values = (rewards + self.gamma * target_q_values) * (1 - terminals)
        predicted_q_values = predicted_q_values.gather(1, actions)
        
        self.online_network.optimizer.zero_grad()
        loss = nn.functional.mse_loss(predicted_q_values, target_q_values).mean()
        loss.backward()
        self.online_network.optimizer.step()        

    def train(self,train_episodes):
        print('Training...')      
        reward_history = []
        best = self.online_network

        for episode_count in range(train_episodes): # Do it for n episodes
            
            episode_start_time = time.time()
            step_count = 0
            self.env_reset() #reset env, get first state, and process it (for the first pick_action)
            terminal=False
            episode_reward=0

            while not terminal: # While not the end of the episode
                action = self.pick_action()
                initial_state = self.frame_stack
                reward, terminal = self.env_step(action)
                self.replay_memory.store_memory(initial_state,  # Pick, see consequences, store 
                                                action,
                                                reward,
                                                terminal,
                                                self.frame_stack)

                self.learn() # Call the smart function

                if step_count % self.target_update_freq == 0:
                    self.update_target_network() # Update Target ?

                episode_reward += reward
                step_count += 1
                
                # ====================Stop Conditions========================
                if step_count >= self.stopping_steps or episode_reward <= self.stopping_reward or (time.time() - episode_start_time)>self.stopping_time:
                    break
                    
            self.update_epsilon()
            reward_history.append(episode_reward)

            current_avg_score = np.mean(reward_history[-10:])

            if(episode_count % 10 == 0): # Update user
                print('ep:{}, Highscore: {}, batch_avg:{}, updated_epsilon:{}'.format(episode_count, self.highscore, current_avg_score, self.epsilon))                   
                print(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec)
                self.save_model_state(best,"huh.pt")
        
            if current_avg_score >= self.highscore: # Save ? 
                self.highscore = current_avg_score
                best = self.online_network #save for later

        self.save_model_state(best,"huh.pt") #actual save
        
    def test(self, test_episodes):
        
        for episode_count in range(test_episodes):
            self.env_reset()
            done=False
            episode_reward = 0
            episode_start_time = time.time()
            
            while not done: #should be fixed? 
                action = self.pick_action()
                reward, done = self.env_step(action)
                episode_reward += reward
                
                if episode_reward <= self.stopping_reward or (time.time() - episode_start_time)>self.stopping_time:
                    break

            print('ep:{}, ep score: {}'.format(episode_count,episode_reward))
        self.env.close()

# implement stop