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

## parsing function for easy running
def parse_args():
        parser = argparse.ArgumentParser(description='Plot results of Simulations')
        parser.add_argument('--env', default="SpaceInvadersDeterministic-v4")
        parser.add_argument('--resultsPath', default="results/")
        parser.add_argument('--play', action='store_true')
        parser.add_argument('--stacks', default=4, type=int, help = 'Amount of frames to stack')
        parser.add_argument('--numSteps', default=10, type=int)
        parser.add_argument('--CNNoption', default='small', type=str, help = 'Choose small or large')
        parser.add_argument('--activation', default=tf.nn.relu)
        parser.add_argument('--nsteps', default=4, help='number of environment steps between training')
        parser.add_argument('--gamma', default=0.95, help='discounted reward factor')
        parser.add_argument('--epsilon', default=0.2, help='Surrogate clipping factor')
        parser.add_argument('--epochs', default = 4, help= 'Number of epochs for training networks')
        parser.add_argument('--learningRate', default = 0.0005, help= 'Starting value for the learning rate for training networks.')
        parser.add_argument('--liverender', default = False, action='store_true')
        parser.add_argument('--nMiniBatch', default = 2, help = 'number of minibatches per trainingepoch')
        parser.add_argument('--loadPath', default = None, help = 'Load existing model')
        parser.add_argument('--saveInterval', default = 5000, help = 'save current network to disk')
        # parser.add_argument('--fc', default=4, type=int)
        args = parser.parse_args()
        return args

#parse arguments and create dict for network options
args=parse_args()
network_args = {}
for item in ['CNNoption','activation','epsilon', 'learningRate', 'epochs', 'nMiniBatch','loadPath']:
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
env = gym.wrappers.Monitor(env, args.resultsPath, force=True)
env = adjustFrame(env)
env = stackFrames(env, args.stacks)
print(env.unwrapped.get_action_meanings())

##create network
tf.reset_default_graph()
sess = tf.Session()
Agent = agent(env, sess, **network_args)


writer = tf.summary.FileWriter(args.resultsPath+"tensorboard", sess.graph)
##reset enviroment
obs = env.reset()

##create list for saving and open file for resultswriting
Rewards = []
EpisodeRewards = []
Actions = []
Observations = []
Values = []
ActionProb = []
resultsFile = open(args.resultsPath+str(time.time())+"results.csv",'a')
resultsFile.write('r,timestep,elapsedtime \n')


##main loop
tStart = tprev = time.time()
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
        resultsFile.write("{}, {}, {} \n".format(sum(EpisodeRewards), timestep, tnow-tprev))
        tprev = tnow
        EpisodeRewards = []

    if (timestep+1) % args.nsteps == 0:
        print("Training Model")
        traintime = time.time()
        Rewards, Observations, Values, Actions, ActionProb= np.asarray(Rewards,dtype=np.float32),  np.asarray(Observations,dtype=np.float32).squeeze(), np.asarray(Values,dtype=np.float32), np.asarray(Actions,dtype=np.int32), np.asarray(ActionProb,dtype=np.float32).reshape((-1,1))
        Advantage, DiscRewards = advantageEST(Rewards,Values,args.gamma)
        Agent.trainNetwork(Observations, Actions, ActionProb, Advantage, DiscRewards)
        Rewards, Actions, Observations, Values, ActionProb = [],[],[],[],[]
        print(time.time()-traintime)

    if timestep % args.saveInterval == 0:
        savePath = args.resultsPath+"checkpoints/"+str(timestep)
        print("Saving model to ",savePath )
        Agent.saveNetwork(savePath)

ttime = time.time()-tStart
print("fps: ", args.numSteps/(ttime))
Agent.saveNetwork(args.resultsPath+"finalModel/final")
resultsFile.close()
env.env.env.env.close()
writer.close()
sess.close()
