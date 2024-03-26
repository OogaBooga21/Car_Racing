from DQN import Network
from DQN_Replay_Mem import Replay_Memory
from DQN_Agent import RL_Agent
import gym

gamma = 0.90
batch_size = 100
memory_size = 300                       
epsilon = 1.0
epsilon_end = 0.02
epsilon_decay = 700
target_update_freq = 64

env = gym.make('CartPole-v1')

replay_memory = Replay_Memory(memory_size)

#========================================================TRAIN======================================================================
# online_network = Network(env)
# target_network = Network(env)
# agent = RL_Agent(env,target_network,online_network,replay_memory,epsilon,epsilon_end,epsilon_decay,batch_size,gamma,target_update_freq)
# agent.fill_memory() # have the replay memory full before doing stuff
# agent.train(1600)
# agent.test(gym.make('CartPole-v1', render_mode='human'),5)


# ========================================================TEST======================================================================
new_online= Network(env)
new_target= Network(env)
agent2 = RL_Agent(env,new_target,new_online,replay_memory,0.0,0.0,0,batch_size,gamma,target_update_freq)
agent2.load_mode_state('/home/oli/SuperMarioWorld/Racing_Car_Implementations/CartPole/Perfect.pt')
agent2.test(gym.make('CartPole-v1', render_mode='human'),50)