import cv2
import time
from collections import deque

class gym_Env_Wrapper:
    stopping_time = 0
    def __init__(self,env,initial_skip_frames, skip_frames, stack_frames, rescale_factor,stopping_reward,stopping_time):
        self.env=env
        self.initial_skip = initial_skip_frames
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        self.episode_history = deque(maxlen=50)
        self.rescale_factor = rescale_factor
        self.stopping_reward = stopping_reward
        self.stopping_time = stopping_time
        self.episode_start_time = 0
        self.episode_reward = 0
        
        dummy_state, _ = self.env.reset()
        img_height = dummy_state.shape[0]
        img_width = dummy_state.shape[1]
        
        self.img_s_h = int(self.rescale_factor * img_height)
        self.img_s_w = int(self.rescale_factor * img_width)
        
    
    def preprocess_state(self,state):
        gray_image = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) / 255.0
        resized_img = cv2.resize(gray_image, (self.img_s_h, self.img_s_w), interpolation=cv2.INTER_AREA)
        return resized_img
    
    def reset(self):
        s, _ = self.env.reset()
        
        for _ in range (self.initial_skip):
            s,_,_,_,_ = self.env.step(0) #no action
        
        s=self.preprocess_state(s)
        
        for _ in range(self.stack_frames):
            self.frame_stack.append(s)
            
        self.episode_start_time = time.time()
        self.episode_reward = 0
        
        return self.frame_stack

    def step(self, action):
        reward = 0
        for _ in range(self.skip_frames):
            s,r,terminal,truncated,info = self.env.step(action)
            self.episode_reward+=r
            reward+=r
            
            s = self.preprocess_state(s)
            self.frame_stack.append(s)
                
            # ====================Stop Conditions========================
            if self.episode_reward < self.stopping_reward or (time.time() - self.episode_start_time)>self.stopping_time:
                terminal = True
            
            if terminal:
                break
            
        return self.frame_stack, reward, terminal
    
    def random_action(self):
        return self.env.action_space.sample()
    
    def current_frame_stack(self):
        return self.frame_stack
    
    def env_state_shape(self):
        return self.stack_frames,self.img_s_h,self.img_s_w
    
    def action_space(self):
        return self.env.action_space
    
    def action_space_size(self):
        return int(self.env.action_space.n)
    