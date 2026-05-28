from time import sleep
from pong_rl_environment import pong_environment
from deepQ_agent import my_agent

env=pong_environment(render=True)

input_neurons=6
output_neurons=3
loadmodel = True    # existing model is loaded
trainme = True     # no training, just play

train_frequency = 10  # Train every x steps
step = 0

agentL=my_agent(input_neurons,output_neurons,loadmodel,trainme,filename="pong_left.keras")
agentR=my_agent(input_neurons,output_neurons,loadmodel,trainme,filename="pong_right.keras")

state, _, _, _ = env.one_step(actionrightpaddle=0)  # Get initial state
while True:
    new_actionL = agentL.get_action(state)
    new_actionR = agentR.get_action(state)

    next_state, rewardR, rewardL, done=env.one_step(actionleftpaddle=new_actionL, actionrightpaddle=new_actionR, human=False)
    agentL.memory.append((state, new_actionL, rewardL, next_state, done))
    agentR.memory.append((state, new_actionR, rewardR, next_state, done))

    if step % train_frequency == 0:
        agentL.train()
        agentR.train()
    
    state = next_state
    step += 1
