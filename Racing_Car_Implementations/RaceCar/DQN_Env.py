import gym

class gymEnvWrapper:
    
    def __init__(self, env, initial_skip_frames, skip_frames, stack_frames, ...):  # Include other preprocessing parameters
        self.env = env
        # ... your preprocessing parameters ...

    def reset(self):
        # Your reset logic with initial frame skipping, etc.  
        ... 
        return processed_state 

    def step(self, action):
        # Your step logic with frame skipping, stacking, reward modifications
        ...
        return processed_state, reward, done, info 