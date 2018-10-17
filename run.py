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
from tensorflow.python import debug as tf_debug

## parsing function for easy running
def parse_args():
        parser = argparse.ArgumentParser(description='Plot results of Simulations')
        parser.add_argument('--env', default="CartPole-v0")#"PongNoFrameskip-v4")
        parser.add_argument('--resultsPath', default=None)
        parser.add_argument('--play', action='store_true')
        parser.add_argument('--stacks', default=4, type=int, help = 'Amount of frames to stack')
        parser.add_argument('--numSteps', default=20, type=int)
        parser.add_argument('--networkOption', default='mlp', type=str, help = 'Choose small or large or mlp')
        parser.add_argument('--activation', default=tf.nn.tanh)
        parser.add_argument('--nsteps', default=10, type=int, help='number of environment steps between training')
        parser.add_argument('--gamma', default=0.99, help='discounted reward factor')
        parser.add_argument('--epsilon', default=0.2, help='Surrogate clipping factor')
        parser.add_argument('--epochs', default = 1, type=int, help= 'Number of epochs for training networks')
        parser.add_argument('--learningRate', default = 3e-4, help= 'Starting value for the learning rate for training networks.')
        parser.add_argument('--liverender', default = False, action='store_true')
        parser.add_argument('--nMiniBatch', default = 2, type=int, help = 'number of minibatches per trainingepoch')
        parser.add_argument('--loadPath', default = None, help = 'Load existing model')
        parser.add_argument('--saveInterval', default = 10000000, type=int, help = 'save current network to disk')
        parser.add_argument('--logInterval', default = 128, type=int, help = 'print Log message')
        parser.add_argument('--cnnStyle', default = 'copy', help = 'copy for 2 CNN and seperate FC layers, shared for shared CNN but seperate FC layers')
        parser.add_argument('--lamda', default = 0.95, help = 'GAE from PPO article')
        parser.add_argument('--c1', default = 0.5, help = 'VF coefficient')
        parser.add_argument('--c2', default = 0.01, help = 'entropy coefficient')
        # parser.add_argument('--fc', default=4, type=int)
        args = parser.parse_args()
        return args

#parse arguments and create dict for network options
args=parse_args()
if not args.resultsPath:
    args.resultsPath = os.path.join("results",args.env+"_"+args.cnnStyle+"_"+args.networkOption)
print(args)
network_args = {}
for item in ['networkOption','activation','epsilon', 'epochs', 'nMiniBatch','loadPath','cnnStyle', 'c1','c2']:
    network_args[item]=args.__dict__[item]
network_args['kernel_initializer'] = tf.ones_initializer()
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
env.seed(0)
# env = gym.wrappers.Monitor(env, args.resultsPath, force=True)
# env = adjustFrame(env)
# env = stackFrames(env, args.stacks)
shape = env.observation_space.shape[0]
print(shape)

##create network
tf.reset_default_graph()
sess = tf.Session()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
Agent = agent(env, sess, **network_args)


writer = tf.summary.FileWriter(os.path.join(args.resultsPath,"tensorboard"), sess.graph)
writer.close()
##reset enviroment
obs = env.reset()

##create list for saving and open file for resultswriting
Rewards = []
EpisodeRewards = []
Actions = []
Observations = []
Values = []
allEpR = []
LogProb = []

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
    print("\n\n t: ", timestep)
    if render:
        env.render()
    # if i %100 ==0:
    #     plt.imshow(obs.reshape((84,84*4),order='F'))
    #     plt.show()
    # obs = obs.reshape((1,84,84,4))

    obs = obs.reshape(1,shape)
    print("obs: ", obs)
    # obs = np.array([-1.,0.,1.]).reshape(-1,3)
    Observations.append(obs)
    # print(obs.shape)
    tt = time.time()
    action, logProb, value, logits = Agent.step(obs,writer,timestep)
    print("action, value, logprob,logits: ", action,value, logProb, logits)
    # print("action: %0.2f, mu: %0.2f, sigma: %0.2f" %(action,mu,sigma))
    # print("l1: ", l1)
    # print("agent step: ",time.time()-tt)
    # print(action, value, actionLogProb)
    # ActionLogProb.append(actionLogProb)
    # action = np.float32(1)
    action = 0 if (timestep %2==0) else 1
    print("selected action: ", action)
    Actions.append(action)
    Values.append(value)
    LogProb.append(logProb)
    tt = time.time()
    obs, reward, done, info = env.step(action)
    # print("obs: ",obs)
    # print(reward)
    # print(Agent.get_Value(obs.reshape(1,3)))
    # print("env step: ", time.time()-tt)
    Rewards.append(reward)
    EpisodeRewards.append(reward)
    # if (timestep+1) % 200 ==0:
    #     done = True
        # print("ep aborted")


    if (timestep+1) % args.nsteps == 0:
        print("Training")
        traintime = time.time()
        Rewards, Observations, Actions, Values, LogProb = np.asarray(Rewards,dtype=np.float32).reshape((-1,1)),  np.asarray(Observations,dtype=np.float32).reshape(-1,shape),  np.asarray(Actions,dtype=np.int32).reshape((-1,1)),  np.asarray(Values,dtype=np.float32).reshape((-1,1)),  np.asarray(LogProb,dtype=np.float32).reshape((-1,1))
        # Advantage = (Advantage - Advantage.mean()) / (Advantage.std() + 1e-8)
        value = Agent.getValue(obs.reshape(1,shape))
        print("v: ", value)
        print("last_obs: ", obs)
        print("Rewards: ", Rewards)
        Advantage, DiscRewards = advantageEST(Rewards, Values, value, args.gamma,args.lamda)
        # print(DiscRewards, Advantage)
        # print(advantageEST(Rewards, Values, value, args.gamma,1))
        # print(Observations)
        # print(Actions)
        # print(DiscRewards)
        # print(DiscRewards, Values, value)
        pLoss, vLoss = Agent.trainNetwork(Observations, Actions, DiscRewards, Values, LogProb, Advantage, args.learningRate)
        Rewards, Actions, Observations, Values, LogProb = [],[],[],[],[]
        # print("Train time: ", time.time()-traintime)

    if done:
        # print("Done")
        tnow = time.time()
        obs = env.reset()
        latestReward = float(sum(EpisodeRewards))
        writeResults("{}, {}, {}".format(latestReward, timestep-tprev, tnow-tStart),'1')
        tprev = timestep
        EpisodeRewards = []
        if len(allEpR)==0:
            allEpR.append(latestReward)
        else:
            allEpR.append(latestReward*0.1+allEpR[-1]*0.9)

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
        print("PolicyLoss: {} \n ValueLoss: {} \n EntropyLoss: {} \n".format(pLoss, vLoss, 0))

ttime = time.time()-tStart
print("fps: ", args.numSteps/(ttime))
# Agent.saveNetwork(os.path.join(args.resultsPath,"finalModel","final.ckpt"))


sess.close()
env.close()
# plt.plot(np.arange(len(allEpR)),allEpR)
# plt.xlabel('Episode')
# plt.ylabel('Moving average')
# plt.show()
