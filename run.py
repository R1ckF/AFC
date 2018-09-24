import gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import cv2
import numpy as np
from gymClass import *
from networkClass import *
import time
import timeit
## parsing function for easy running
def parse_args():
        parser = argparse.ArgumentParser(description='Plot results of Simulations')
        parser.add_argument('--env', default="SpaceInvadersDeterministic-v4")
        parser.add_argument('--resultsPath', default="results/")
        parser.add_argument('--play', action='store_true')
        parser.add_argument('--stacks', default=4, type=int)
        parser.add_argument('--numSteps', default=10, type=int)
        parser.add_argument('--CNNoption', default='small', type=str)
        parser.add_argument('--activation', default=tf.nn.elu)
        parser.add_argument('--minibatch', default=4)
        parser.add_argument('--gamma', default=0.9, help='discouted reward factor')
        # parser.add_argument('--fc', default=4, type=int)
        args = parser.parse_args()
        return args

#parse arguments and create dict for network options
args=parse_args()
network_args = {}
for item in ['CNNoption','activation']:
    network_args[item]=args.__dict__[item]

#create environement
env = gym.make(args.env)
env = gym.wrappers.Monitor(env, args.resultsPath, force=True)
env = adjustFrame(env)
env = stackFrames(env, args.stacks)
print(env.unwrapped.get_action_meanings())

##create network
sess = tf.Session()
Agent = agent(env, sess, **network_args)
writer = tf.summary.FileWriter("output", sess.graph)
##reset enviroment
obs = env.reset()

##create list for saving
Rewards = []
Actions = []
Observations = []
Values = []

##main loop
tStart = time.time()
for i in range(args.numSteps):
    env.render()
    # if i %100 ==0:
    #     plt.imshow(obs.reshape((84,84*4),order='F'))
    #     plt.show()

    obs = obs.reshape((1,84,84,4))
    Observations.append(obs)
    action, value, actionSpaceProb = Agent.step(obs)
    # print(action)
    print(action)
    Actions.append(action)
    Values.append(value)
    obs, reward, done, info = env.step(action)
    Rewards.append(reward)
    if done:
        obs = env.reset()

    if i % args.minibatch == 0:
        dReward = discountRewards(Rewards,Values,args.gamma)
        


ttime = time.time()-tStart
print(max(Rewards))
print(Values)
print("fps: ", args.numSteps/(ttime))
env.env.env.env.close()
writer.close()
sess.close()
