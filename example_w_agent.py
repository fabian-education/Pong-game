from time import sleep
from pong_rl_environment import pong_environment
from deepQ_agent import my_agent

env=pong_environment(render=True)

input_neurons=6
output_neurons=3
loadmodel = False    # existing model is loaded
trainme = True     # training enabled (exploration + learning)

train_frequency = 10  # Train every x steps
step = 0

agent=my_agent(input_neurons,output_neurons,loadmodel,trainme, filename="pong_single_agent.keras")

state, _, _, _ = env.one_step(actionrightpaddle=0)  # Get initial state
while True:
    new_action = agent.get_action(state)
    next_state, reward, rewardleft, done=env.one_step(actionrightpaddle=new_action)
    agent.memory.append((state, new_action, reward, next_state, done))
    
    if step % train_frequency == 0:
        agent.train()
    
    state = next_state
    step += 1


