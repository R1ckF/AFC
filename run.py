import gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import cv2
import numpy as np
from gymClass import *
from networkClass import *
## parsing function for easy running
def parse_args():
        parser = argparse.ArgumentParser(description='Plot results of Simulations')
        parser.add_argument('--env', default="SpaceInvaders-v0")
        parser.add_argument('--resultsPath', default="results/")
        parser.add_argument('--play', action='store_true')
        parser.add_argument('--stacks', default=4, type=int)
        parser.add_argument('--numSteps', default=3, type=int)
        parser.add_argument('--cnn', default=4, type=int)
        parser.add_argument('--fc', default=4, type=int)
        args = parser.parse_args()
        return args

#parse arguments
args=parse_args()

##create environement
env = gym.make(args.env)
env = gym.wrappers.Monitor(env, args.resultsPath, force=True)
env = adjustFrame(env)
env = stackFrames(env, args.stacks)

##create network
sess = tf.Session()

runner = agent(env, sess, args.cnn, args.fc)


##reset enviroment
obs = env.reset()

##create list for saving
Rewards = []

##main loop
for _ in range(args.numSteps):
    env.render()
    action = runner.step(obs)
    print(action)
    # obs, reward, done, info = env.step(action)
    # if done: obs = env.reset()
    # Rewards.append(reward)
print(action.dtype)
env.env.env.env.close()
sess.close()
