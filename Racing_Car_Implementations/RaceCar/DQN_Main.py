from DQN_Agent import RL_Agent
import gym
import numpy as np

gamma = 0.95
batch_size = 1024
memory_size = 60000            
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 1000 # 10% - 20% of training episodes
target_update_freq = 100
stopping_steps = np.inf # if 0 we dont take it into account
stopping_reward = -15 #if -inf we dont take this into account
stopping_time = np.inf #if 0 we dont take this into account (seconds)
initial_skip_frames  = 50
skip_frames = 4
stack_frames = 4
rescale_factor = 1.0

env = gym.make('CarRacing-v2' ,continuous=False)

#========================================================TRAIN=====================================================================

agent = RL_Agent(env, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq, rescale_factor,
                 stopping_reward,stopping_time, stopping_steps,
                 initial_skip_frames,skip_frames,stack_frames)

agent.train(5000)

#========================================================TEST======================================================================
memory_size = 1
env_test = gym.make('CarRacing-v2',domain_randomize=False,continuous=False,render_mode='human')
agent_test = RL_Agent(env_test, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq, rescale_factor,
                 stopping_reward,stopping_time, stopping_steps,
                 initial_skip_frames,skip_frames,stack_frames)
agent_test.load_mode_state('huh.pt')
agent_test.test(10)