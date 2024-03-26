from DQN_Agent import RL_Agent
import gym
import numpy as np

gamma = 0.90
batch_size = 64 #
memory_size = 392                      
epsilon = 1.0
epsilon_end = 0.02
epsilon_decay = 20 #700 ? 
target_update_freq = 32
stopping_steps = np.inf # if 0 we dont take it into account
stopping_reward = -30 #if -inf we dont take this into account
stopping_time = 60 #if 0 we dont take this into account (seconds)
initial_skip_frames  = 55
skip_frames = 4
stack_frames = 4
rescale_factor = 0.85

env = gym.make('CarRacing-v2' ,continuous=False)

#========================================================TRAIN======================================================================

# agent = RL_Agent(env, memory_size,
#                  epsilon,epsilon_end,epsilon_decay,
#                  batch_size,gamma,target_update_freq, rescale_factor,
#                  stopping_reward,stopping_time, stopping_steps,
#                  initial_skip_frames,skip_frames,stack_frames)

# agent.train(200)

#========================================================TEST======================================================================
memory_size = 70
env_test = gym.make('CarRacing-v2',domain_randomize=False,continuous=False,render_mode='human')
agent_test = RL_Agent(env_test, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq, rescale_factor,
                 stopping_reward,stopping_time, stopping_steps,
                 initial_skip_frames,skip_frames,stack_frames)
agent_test.load_mode_state('huh.pt')
agent_test.test(10)