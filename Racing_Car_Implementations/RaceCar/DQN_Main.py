from DQN_Agent import RL_Agent
from DQN_Env import gym_Env_Wrapper as gym_Wrapper
import gym
import numpy as np

gamma = 0.95
batch_size = 32
memory_size = 10000        
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 1000 # 50-30% ?
target_update_freq = 100

stopping_reward = -50
stopping_time = 10
initial_skip_frames  = 50
skip_frames = 4
stack_frames = 4
rescale_factor = 0.8

car_racer = gym.make('CarRacing-v2',continuous=False)

env = gym_Wrapper(car_racer,initial_skip_frames,skip_frames,stack_frames,
                 rescale_factor,stopping_reward,stopping_time)
#========================================================TRAIN=====================================================================

agent = RL_Agent(env, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq)

agent.train(500)
env.stopping_time+=10
agent.train(500)
env.stopping_time+=20
agent.train(500)
env.stopping_time+=30
agent.train(500)

agent.test(5)

#========================================================TEST======================================================================
# memory_size = 1
# env_test = gym.make('CarRacing-v2',domain_randomize=False,continuous=False,render_mode='human')
# agent_test = RL_Agent(env_test, memory_size,
#                  epsilon,epsilon_end,epsilon_decay,
#                  batch_size,gamma,target_update_freq, rescale_factor,
#                  stopping_reward,stopping_time, stopping_steps,
#                  initial_skip_frames,skip_frames,stack_frames)
# agent_test.load_mode_state('huh.pt')
# agent_test.test(10)