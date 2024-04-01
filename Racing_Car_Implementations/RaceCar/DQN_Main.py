from DQN_Agent import RL_Agent
from DQN_Env import gym_Env_Wrapper as gym_Wrapper
import gym
import numpy as np

gamma = 0.95
batch_size = 64
memory_size = 10000      
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 1500 # 50-30% ?
target_update_freq = 100

stopping_reward = -50
stopping_time = 20 #mostly used for test
stopping_steps = 200 #mostly used for training
initial_skip_frames  = 50
skip_frames = 4
stack_frames = 4
rescale_factor = 1.0

# #========================================================TRAIN=====================================================================
car_racer = gym.make('CarRacing-v2',continuous=False)

env = gym_Wrapper(car_racer,initial_skip_frames,skip_frames,stack_frames,
                 rescale_factor,stopping_reward,stopping_time,stopping_steps)

agent = RL_Agent(env, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq)

agent.train(1250)
# env.stopping_time+=10
env.stopping_steps+=200
agent.train(500)
# env.stopping_time+=20
env.stopping_steps+=300
agent.train(500)
# env.stopping_time+=30
env.stopping_steps+=300
agent.train(500)

agent.test(5)

#========================================================TEST======================================================================
memory_size = 1000
env_test = gym.make('CarRacing-v2',domain_randomize=False,continuous=False,render_mode='human')

env = gym_Wrapper(env_test,initial_skip_frames,skip_frames,stack_frames,
                 rescale_factor,stopping_reward,stopping_time,stopping_steps)

agent_test = RL_Agent(env, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq)

agent_test.load_mode_state('huh.pt')
agent_test.test(10)
