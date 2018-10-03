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
import pickle
import os

## parsing function for easy running
def parse_args():
        parser = argparse.ArgumentParser(description='Plot results of Simulations')
        parser.add_argument('--env', default="PongNoFrameskip-v4")
        parser.add_argument('--resultsPath', default=None)
        parser.add_argument('--play', action='store_true')
        parser.add_argument('--stacks', default=4, type=int, help = 'Amount of frames to stack')
        parser.add_argument('--numSteps', default=5000000, type=int)
        parser.add_argument('--CNNoption', default='small', type=str, help = 'Choose small or large')
        parser.add_argument('--activation', default=tf.nn.relu)
        parser.add_argument('--nsteps', default=128, type=int, help='number of environment steps between training')
        parser.add_argument('--gamma', default=0.99, help='discounted reward factor')
        parser.add_argument('--epsilon', default=0.2, help='Surrogate clipping factor')
        parser.add_argument('--epochs', default = 3, type=int, help= 'Number of epochs for training networks')
        parser.add_argument('--learningRate', default = 2.5e-4, help= 'Starting value for the learning rate for training networks.')
        parser.add_argument('--liverender', default = False, action='store_true')
        parser.add_argument('--nMiniBatch', default = 4, type=int, help = 'number of minibatches per trainingepoch')
        parser.add_argument('--loadPath', default = None, help = 'Load existing model')
        parser.add_argument('--saveInterval', default = 100000, type=int, help = 'save current network to disk')
        parser.add_argument('--logInterval', default = 10000, type=int, help = 'print Log message')
        parser.add_argument('--cnnStyle', default = 'copy', help = 'copy for 2 CNN and seperate FC layers, shared for shared CNN but seperate FC layers')
        parser.add_argument('--lamda', default = 0.95, help = 'GAE from PPO article')
        parser.add_argument('--c1', default = 1, help = 'VF coefficient')
        parser.add_argument('--c2', default = 0.01, help = 'entropy coefficient')
        # parser.add_argument('--fc', default=4, type=int)
        args = parser.parse_args()
        return args

#parse arguments and create dict for network options
args=parse_args()
if not args.resultsPath:
    args.resultsPath = os.path.join("results",args.env+"_"+args.cnnStyle+"_"+args.CNNoption)
print(args)
network_args = {}
for item in ['CNNoption','activation','epsilon', 'learningRate', 'epochs', 'nMiniBatch','loadPath','cnnStyle', 'c1','c2']:
    network_args[item]=args.__dict__[item]
render = args.liverender
assert ((args.nsteps/args.nMiniBatch) % 1 == 0)
# rewards = np.ones(10)
# values = (np.ones(10)-np.array([0.1, 0.1,0.1,0.1,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1]))*10
#
# advDR,DR = advantageDR(rewards,values,0.5)
# print(advDR)
# print(DR)
# advEST = advantageEST(rewards,values,0.5)
# print(advEST)
# print(advEST+values)
# stolen ,dr = stolen(rewards,values,0.5)
# print(stolen)
# print(dr)

#create environement
env = gym.make(args.env)
# env = gym.wrappers.Monitor(env, args.resultsPath, force=True)
env = adjustFrame(env)
env = stackFrames(env, args.stacks)


##create network
tf.reset_default_graph()
sess = tf.Session()
Agent = agent(env, sess, **network_args)


writer = tf.summary.FileWriter(os.path.join(args.resultsPath,"tensorboard"), sess.graph)
##reset enviroment
obs = env.reset()

##create list for saving and open file for resultswriting
Rewards = []
EpisodeRewards = []
Actions = []
Observations = []
Values = []
ActionProb = []
def writeResults(message,file):
    resultsFile = open(os.path.join(args.resultsPath,file+".results.csv"),'a')
    resultsFile.write(message+'\n')
    resultsFile.close()

writeResults('r,timestep,elapsedtime','1')


##main loop
tStart = time.time()
tprev = 0
latestReward = 0
for timestep in range(args.numSteps):
    if render:
        env.render()
    # if i %100 ==0:
    #     plt.imshow(obs.reshape((84,84*4),order='F'))
    #     plt.show()
    obs = obs.reshape((1,84,84,4))
    Observations.append(obs)
    action, value, actionProb = Agent.step(obs)
    ActionProb.append(actionProb)
    Actions.append(action)
    Values.append(value)
    obs, reward, done, info = env.step(action)
    Rewards.append(reward)
    EpisodeRewards.append(reward)

    if done:
        tnow = time.time()
        obs = env.reset()
        latestReward = sum(EpisodeRewards)
        writeResults("{}, {}, {}".format(latestReward, timestep-tprev, tnow-tStart),'1')
        tprev = timestep
        EpisodeRewards = []

    if (timestep+1) % args.nsteps == 0:
        # traintime = time.time()
        Rewards, Observations, Values, Actions, ActionProb= np.asarray(Rewards,dtype=np.float32).reshape((-1,1)),  np.asarray(Observations,dtype=np.float32).squeeze(), np.asarray(Values,dtype=np.float32).reshape((-1,1)), np.asarray(Actions,dtype=np.int32).reshape((-1,1)), np.asarray(ActionProb,dtype=np.float32).reshape((-1,1))
        Advantage, DiscRewards = advantageEST(Rewards,Values,args.gamma,args.lamda)
        lClip, lVF, entropy = Agent.trainNetwork(Observations, Actions, ActionProb, Advantage, DiscRewards)
        Rewards, Actions, Observations, Values, ActionProb = [],[],[],[],[]
        # print(time.time()-traintime)

    if (timestep+1) % args.saveInterval == 0:
        savePath = os.path.join(args.resultsPath,"checkpoints"+str(timestep)+".ckpt")
        Agent.saveNetwork(savePath)
        print("Saving model to ",savePath )

    if (timestep+1) % args.logInterval == 0:
        # print(time.time(), tStart, timestep, args.numSteps)
        esttime = (time.time()-tStart)/float(timestep)*(args.numSteps-timestep)
        esttime = time.strftime("%H:%M:%S", (time.gmtime(esttime)))
        print("Latest reward: ", latestReward)
        print("Estimated time remaining: ", esttime)
        print("Update {} of {}".format((timestep+1)/args.logInterval, args.numSteps/args.logInterval))
        print("PolicyLoss: {} \n ValueLoss: {} \n EntropyLoss: {} \n".format(-lClip, lVF, entropy))

ttime = time.time()-tStart
print("fps: ", args.numSteps/(ttime))
Agent.saveNetwork(os.path.join(args.resultsPath,"finalModel","final.ckpt"))
env.env.env.env.close()
writer.close()
sess.close()
