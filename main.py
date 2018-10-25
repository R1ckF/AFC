import gym
import argparse
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from networkClassFC import *
import time
import os

"""
Main function used for running each training run
Takes the variable parameters that are testes as inputs
"""
def main(play=False, nsteps=128, clippingFactor=lambda f: 0.2*f, epochs=4, nMiniBatch=4, learningRate=lambda f: f * 2.0e-4, activation=tf.nn.tanh, numNodes=64, numLayers=2, seed=0):

    ##define some constants similar for each training run
    env = "CartPole-v0"
    saveInterval = 30000
    logInterval = 10000
    numSteps = 20000
    Lamda = 0.95
    gamma = 0.99
    networkStyle='copy'
    c1 = 0.5
    loadPath = None

    resultsPath = os.path.join("results",str(numNodes)+"_"+str(numLayers))

    network_args = {'networkOption': 'fc'}
    for item in ['activation', 'epochs', 'nMiniBatch','loadPath','networkStyle', 'c1', 'numNodes', 'numLayers']:
        network_args[item]=locals()[item]
    print(network_args)
    ## ensure values match to avoid errors later on
    assert ((nsteps/nMiniBatch) % 1 == 0)


    #create environement
    env = gym.make(env)
    env.seed(seed)
    ob_shape = list(env.observation_space.shape)
    print(ob_shape)

    ##create network
    tf.reset_default_graph()
    sess = tf.Session()
    Agent = agent(env, sess, **network_args)

    ## create tensorboard file
    writer = tf.summary.FileWriter(os.path.join(resultsPath,"tensorboard",str(seed)), sess.graph)
    writer.close()

    ##reset enviroment
    obs = env.reset()
    obs = obs.reshape([1]+ob_shape)


    ##create list for saving
    Rewards = []
    EpisodeRewards = []
    Actions = []
    Observations = []
    Values = []
    allEpR = []
    LogProb = []
    Dones = []
    Timesteps = []
    ElapsedTime = []

    ## main loop
    tStart = time.time()
    for timestep in range(numSteps):
        Observations.append(obs)
        #input observation and get the action, logarthmic probabilty and value from the agent
        action, logProb, value = Agent.step(obs)
        # store in lists
        Actions.append(action)
        Values.append(value)
        LogProb.append(logProb)
        # apply action to environment and retreive next observation
        obs, reward, done, info = env.step(action)
        obs= obs.reshape([1]+ob_shape)
        #store in lists
        Dones.append(done)
        Rewards.append(reward)
        EpisodeRewards.append(reward)


        if (timestep+1) % nsteps == 0:

            lr = learningRate(1-timestep/numSteps) # calc current learning rate
            epsilon = clippingFactor(1-timestep/numSteps) #calc current epsilon
            Dones, Rewards, Observations, Actions, Values, LogProb = np.asarray(Dones), np.asarray(Rewards,dtype=np.float32),  np.asarray(Observations,dtype=np.float32).reshape([nsteps]+ob_shape),  np.asarray(Actions,dtype=np.int32),  np.asarray(Values,dtype=np.float32),  np.asarray(LogProb,dtype=np.float32)
            value = Agent.getValue(obs) # get value from latest observation
            Advantage, DiscRewards = advantageEST(Rewards, Values, Dones, value, gamma,Lamda) #calculate advantange and discounted rewards according to the method in the article
            pLoss, vLoss = Agent.trainNetwork(Observations, Actions, DiscRewards, Values, LogProb, Advantage, lr, epsilon) # train network
            Rewards, Actions, Observations, Values, LogProb, Dones = [],[],[],[],[],[] # create new lists for next batch

        if done: # current episode is finished and needs reset. Also used as a checkpoint for saving intermediate results
            tnow = time.time()
            obs = env.reset()
            obs= obs.reshape([1]+ob_shape)
            latestReward = float(sum(EpisodeRewards))
            EpisodeRewards = []
            allEpR.append(latestReward)
            Timesteps.append(timestep)
            ElapsedTime.append(tnow-tStart)

        if (timestep+1) % saveInterval == 0: # save current network parameters to disk
            savePath = os.path.join(resultsPath,"checkpoints"+str(timestep)+".ckpt")
            Agent.saveNetwork(savePath)
            print("Saving model to ",savePath )

        if (timestep+1) % logInterval == 0: # print summary to screen
            esttime = (time.time()-tStart)/float(timestep)*(numSteps-timestep)
            esttime = time.strftime("%H:%M:%S", (time.gmtime(esttime)))
            print("Latest reward: ", latestReward)
            print("Estimated time remaining: ", esttime)
            print("Update {} of {}".format((timestep+1)/logInterval, numSteps/logInterval))
            print("PolicyLoss: {} \n ValueLoss: {} \n EntropyLoss: {} \n".format(pLoss, vLoss, 0))

    ttime = time.time()-tStart
    print("fps: ", numSteps/(ttime))
    Agent.saveNetwork(os.path.join(resultsPath,"finalModel","final.ckpt"))


    if play: # provide short visual render of resulting network
        for _ in range(100):
            obs = env.reset()
            done = True
            while done:
                env.render()
                time.sleep(0.1)
                obs= obs.reshape([1]+ob_shape)
                action, logProb, value = Agent.step(obs)
                obs, reward, done, info = env.step(action)

    sess.close()
    env.close()

    return allEpR, Timesteps, ElapsedTime

if __name__ == "__main__":
    allEpR, Timesteps, ElapsedTime = main()
    plt.figure()
    plt.plot(np.arange(len(allEpR)),allEpR)
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.figure()
    plt.plot(Timesteps,allEpR)
    plt.xlabel('Timestep')
    plt.ylabel('Reward')

    plt.figure()
    plt.plot(ElapsedTime,allEpR)
    plt.xlabel('Time [s]')
    plt.ylabel('Reward')

    plt.show()
