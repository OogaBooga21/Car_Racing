import cv2
import time
from collections import deque
import numpy as np
import cv2
from PIL import Image
import pygame

class gym_Env_Wrapper:
    stopping_time = 0
    stopping_reward = 0
    stopping_steps = 0
    
    window_size = (150,150)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("Grayscale Image")
    
    def __init__(self,env,initial_skip_frames, skip_frames, stack_frames, rescale_factor,stopping_reward,stopping_time,stopping_steps):
        self.env=env
        self.initial_skip = initial_skip_frames
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        self.rescale_factor = rescale_factor
        self.stopping_reward = stopping_reward
        self.stopping_time = stopping_time
        self.stopping_steps = stopping_steps
        self.episode_start_time = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.bad_step_counter = 0
        
        dummy_state, _ = self.env.reset()
        img_height = dummy_state.shape[0]
        img_width = dummy_state.shape[1]
        
        self.img_s_h = int(self.rescale_factor * img_height)
        self.img_s_w = int(self.rescale_factor * img_width)
        
    
    def preprocess_state(self,state):
        gray_image = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) / 255.0
        if self.rescale_factor != 1.0:
            gray_image = cv2.resize(gray_image, (self.img_s_h, self.img_s_w), interpolation=cv2.INTER_AREA)
        
        gray_image  = np.where(gray_image > 0.6, 0.0, gray_image)
        gray_image  = np.where(gray_image < 0.35, 0.0, gray_image)
        
        self.display_image(gray_image)
        # gray_image  = np.where(gray_image > 0.3, 1.0, gray_image) #optional, makes the tarck completely white
        return gray_image #experimental image processing

    def reset(self):
        s, _ = self.env.reset()
        
        for _ in range (self.initial_skip):
            s,_,_,_,_ = self.env.step(0) #no action
        
        s=self.preprocess_state(s)
        
        for _ in range(self.stack_frames):
            self.frame_stack.append(s)
            
        self.episode_start_time = time.time()
        self.episode_reward = 0
        self.episode_steps = 0
        self.bad_step_counter = 0
        
        return self.frame_stack

    def step(self, action):
        stack_reward = 0
        for _ in range(self.skip_frames):
            s,r,terminal,truncated,info = self.env.step(action)
                        
            stack_reward+=r
            
            s = self.preprocess_state(s)
            self.frame_stack.append(s)
                
            # ====================Stop Conditions========================
            if self.episode_reward < self.stopping_reward or (time.time() - self.episode_start_time)>self.stopping_time or terminal == True:
                terminal = True
                break
            
        self.episode_steps +=1
        if self.episode_steps > self.stopping_steps:
                terminal = True
    # ===========================Reward Shaping ==========================================
        if stack_reward < 0:
            stack_reward = 0.02*self.bad_step_counter*stack_reward
            self.bad_step_counter += 1 #1 will not even be affected 2 will barelly be
        else:
            self.bad_step_counter = 0
            
        if action == 3:
            stack_reward += 0.05 #Encourage it to move forward, but not enough to be always worth it
        
        self.episode_reward += stack_reward
        
        return self.frame_stack, stack_reward, terminal
    
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
    
    def display_image(self, image_data):
        image_data = (image_data * 255).astype(np.uint8)
        image_data = np.transpose(image_data, (1, 0))
        image_surface = pygame.Surface(image_data.shape, depth=8)
        pygame.surfarray.blit_array(image_surface, image_data)
        # grayscale_palette = [(i, i, i) for i in range(256)]  
        # image_surface.set_palette(grayscale_palette)
        # for grayscale but its slower
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.VIDEORESIZE:
                self.window_size = (event.w, event.h)
                self.screen = pygame.display.set_mode(self.window_size, pygame.RESIZABLE)

        image_surface = pygame.transform.scale(image_surface, self.window_size)
        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()