from DQN_Agent import RL_Agent
from DQN_Env import gym_Env_Wrapper as gym_Wrapper
import gym

# Example Usage:
gamma = 0.9
batch_size = 256
memory_size = 3000     
epsilon = 1.0
epsilon_end = 0.01
epsilon_decay = 4500 # 50-30% ?
target_update_freq = 100

stopping_reward = -200.0
stopping_time = 100 #mostly used for test
stopping_steps = 1000 #mostly used for training (wrapper steps not env steps)
initial_skip_frames  = 50
skip_frames = 5
stack_frames = 10
rescale_factor = 0.30
mini_render = True
# #========================================================TRAIN=====================================================================
car_racer = gym.make('CarRacing-v2',domain_randomize= False,continuous=False)

env = gym_Wrapper(car_racer,mini_render,initial_skip_frames,skip_frames,stack_frames,
                 rescale_factor,stopping_reward,stopping_time,stopping_steps)

agent = RL_Agent(env, memory_size,
                 epsilon,epsilon_end,epsilon_decay,
                 batch_size,gamma,target_update_freq)

agent.train(6000)
# env.stopping_time+=10
# env.stopping_steps+=100
# agent.train(2000)
# # env.stopping_time+=20
# env.stopping_steps+=200
# agent.train(2000)
# # env.stopping_time+=30
# env.stopping_steps+=300
# agent.train(500)
agent.test(5)

#========================================================TEST======================================================================
memory_size = 2
mini_render = False
stopping_time = 120 #mostly used for test
stopping_steps = 1000 #mostly used for training (wrapper steps not env steps)
test = True

env_test = gym.make('CarRacing-v2',domain_randomize=False,continuous=False, render_mode='human')
env = gym_Wrapper(env_test,mini_render,initial_skip_frames,skip_frames,stack_frames,
                 rescale_factor,stopping_reward,stopping_time,stopping_steps)

agent_test = RL_Agent(env, memory_size,
                 0,0,0,
                 batch_size,gamma,target_update_freq)

agent_test.load_model_state('bestcomp.pt',test)
agent_test.test(10)