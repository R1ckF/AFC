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
        parser.add_argument('--numSteps', default=1000, type=int)
        parser.add_argument('--CNNoption', default='small', type=str)
        # parser.add_argument('--fc', default=4, type=int)
        args = parser.parse_args()
        return args

#parse arguments
args=parse_args()

##create environement
env = gym.make(args.env)
env = gym.wrappers.Monitor(env, args.resultsPath, force=True)
env = adjustFrame(env)
env = stackFrames(env, args.stacks)
print(env.unwrapped.get_action_meanings())

##create network
sess = tf.Session()
runner = agent(env, sess, args.CNNoption)
writer = tf.summary.FileWriter("output", sess.graph)

##reset enviroment
obs = env.reset()

##create list for saving
Rewards = []

##main loop

t = time.time()
d = 0
for i in range(args.numSteps):
    env.render()
    lt = time.time()
    # if i %100 ==0:
    #     plt.imshow(obs.reshape((84,84*4),order='F'))
    #     plt.show()
    action = runner.step(obs.reshape((1,84,84,4)))
    # print(time.time()-lt)
    # print(action)
    # action = np.random.random(size=(6))
    action = np.argmax(action)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        d +=1
    Rewards.append(action)
    # print(obs.shape)
ttime = time.time()-t
print('number of played eps: ', d)
print(args.numSteps)
print("fps: ", args.numSteps/(ttime))
env.env.env.env.close()
writer.close()
sess.close()
