from main import *
import pickle
import tensorflow as tf
import os
import time

START = time.time()
play=False
nstepsL=[64,128, 256, 512]
epsilonL=[lambda f: 0.1, lambda f: 0.2, lambda f: 0.3]
epochsL=[1,4,8,16]
nMiniBatchL=[1,2,4,8]
learningRateL=[lambda f: 3.0e-4]
activationL=[tf.nn.tanh]

play=False
nstepsL=[128]
epsilonL=[lambda f: 0.2]
epochsL=[1,4,10]
nMiniBatchL=[1,4,8]
learningRateL=[lambda f: 2.0e-4]
activationL=[tf.nn.tanh]

numNodesL=[16,32,64,128]
numLayersL= [1,2,3,4]
seedL=[i for i in range(10)]

def saveVariables(filename, variables):
    with open(os.path.join("results",filename), 'wb') as f:
        pickle.dump(variables, f)

for nsteps in nstepsL:
    for clippingFactor in epsilonL:
        for epochs in epochsL:
            for nMiniBatch in nMiniBatchL:
                for learningRate in learningRateL:
                    for activation in activationL:
                        for numNodes in numNodesL:
                            for numLayers in numLayersL:
                                for seed in seedL:
                                    runParam={}
                                    for item in ["nsteps", "clippingFactor", "epochs", "nMiniBatch", "learningRate", "activation", "numNodes", "numLayers", "seed"]:
                                        runParam[item]=locals()[item]
                                    allEpR, Timesteps, ElapsedTime = main(**runParam)
                                    saveVariables(str(nsteps)+"_"+str(epochs)+"_"+str(nMiniBatch)+"_"+str(numLayers)+"_"+str(numNodes)+"_"+str(seed),[allEpR, Timesteps, ElapsedTime])


print("Done in %02f seconds" %(time.time()-START))
# allEpR, Timesteps, ElapsedTime = main(**runParam)

# a = tf.nn.tanh
# print(i*2/60)

    # def parse_args():
    #         parser = argparse.ArgumentParser(description='Plot results of Simulations')
    #         parser.add_argument('--env', default="CartPole-v0")
    #         parser.add_argument('--play', default= play, action='store_true', help='add to get a short visual sample of the result')
    #         parser.add_argument('--numSteps', default=10000, type=int)
    #         parser.add_argument('--nsteps', default=128, type=int, help='number of environment steps between training')
    #         parser.add_argument('--gamma', default=0.99, help='discounted reward factor')
    #         parser.add_argument('--epsilon', default=0.2, help='Surrogate clipping factor')
    #         parser.add_argument('--epochs', default = 4, type=int, help= 'Number of epochs for training networks')
    #         parser.add_argument('--learningRate', default = lambda f: f * 2.5e-4, help= 'Starting value for the learning rate for training networks.')
    #         parser.add_argument('--nMiniBatch', default = 4, type=int, help = 'number of minibatches per trainingepoch')
    #         parser.add_argument('--loadPath', default =None)# "results/CartPole-v0_copy_fc/finalModel/final.ckpt", help = 'Load existing model')
    #         parser.add_argument('--saveInterval', default = 1000, type=int, help = 'save current network to disk')
    #         parser.add_argument('--logInterval', default = 1000, type=int, help = 'print Log message')
    #         parser.add_argument('--networkStyle', default = 'copy', help = 'copy for seperate FC layers, shared for shared network but seperate value and action layers')
    #         parser.add_argument('--activation', default=tf.nn.tanh)
    #         parser.add_argument('--lamda', default = 0.95, help = 'GAE from PPO article')
    #         parser.add_argument('--c1', default = 1, help = 'VF coefficient from article')
    #         parser.add_argument('--seed', default = 0, help = 'seed for gym env')
    #         args = parser.parse_args()
    #         return args

    #parse arguments and create dict for network options
