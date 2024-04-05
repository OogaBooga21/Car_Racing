from DQN_Agent import RL_Agent
from DQN_Env import gym_Env_Wrapper as gym_Wrapper
import gym
import numpy as np
from PIL import Image
import numpy as np  # Assuming your state is a NumPy array

# Example Usage:
gamma = 0.85
batch_size = 256
memory_size = 3000     
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 4000 # 50-30% ?
target_update_freq = 250

stopping_reward = -200
stopping_time = 100 #mostly used for test
stopping_steps = 300 #mostly used for training (wrapper steps not env steps)
initial_skip_frames  = 50
skip_frames = 4
stack_frames = 8
rescale_factor = 0.25
 
# #========================================================TRAIN=====================================================================
car_racer = gym.make('CarRacing-v2',domain_randomize= False,continuous=False)

env = gym_Wrapper(car_racer,initial_skip_frames,skip_frames,stack_frames,
                 rescale_factor,stopping_reward,stopping_time,stopping_steps)

#=============test, just to see how the image looks===================
# stack = env.reset()
# for i in range(200):
#     stack,reward,_ = env.step(3)

# brakpoint

agent = RL_Agent(env, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq)

agent.train(8000)
# env.stopping_time+=10
env.stopping_steps+=100
agent.train(2000)
# env.stopping_time+=20
env.stopping_steps+=200
agent.train(2000)
# env.stopping_time+=30
env.stopping_steps+=300
agent.train(500)

agent.test(5)

#========================================================TEST======================================================================
# memory_size = 2
# env_test = gym.make('CarRacing-v2',domain_randomize=False,continuous=False)

# env = gym_Wrapper(env_test,initial_skip_frames,skip_frames,stack_frames,
#                  rescale_factor,stopping_reward,stopping_time,stopping_steps)

# agent_test = RL_Agent(env, memory_size,
#                  epsilon,epsilon_end,epsilon_decay,
#                  batch_size,gamma,target_update_freq)

# agent_test.load_mode_state('best.pt')
# agent_test.test(10)