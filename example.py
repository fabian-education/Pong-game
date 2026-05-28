from time import sleep
import random
from pong_rl_environment import pong_environment
#from deepQ_agent import my_agent

env=pong_environment(render=True)

#input_neurons=8
#output_neurons=3
#loadmodel = False    # existing model is loaded
#trainme = True     # training enabled (exploration + learning)

#agent=my_agent(input_neurons,output_neurons,loadmodel,trainme)

while True:
    for i in range(random.randint(100,300)):
        positiondata, reward, rewardleft, done=env.one_step(actionrightpaddle=random.randint(0,2))
        sleep(0.05)


