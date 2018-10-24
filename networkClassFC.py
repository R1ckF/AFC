import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import gym

class network:
    """
    This class is used as a framework that can be called to create actor / critic networks.
    env: gym environment (needed to obtain outputspace of network to match input of env)
    networkOption: the network that is selected (only fc is used as cnn's took to long to trainNetwork
    sess: tensorflow session used to run the network
    """
    def __init__(self,env, networkOption, sess):
        self.env = env
        self.networkOption = networkOption

        ## function that creates a convolutional layer. Just reorganised it a bit for easier typing
    def conv(self, input, outputDepth, kernelsize, stride, padding="valid", name = 'Conv', **network_args):
        return tf.layers.conv2d(
          inputs=input,
          filters=outputDepth,
          kernel_size=[kernelsize, kernelsize],strides=(stride,stride),padding=padding, name = name, **network_args)

          ##Function that creates a fully connected layer. Just reorganised it a bit for easier typing
    def fc(self, input, numOutputs, name = None, **network_args):
        return tf.layers.dense(
            input,
            numOutputs,
            name = name, **network_args)

        ## building cnnsmall consisting of 2 convolutional layers and 1 FC layer
    def cnnSmall(self, observationPH, **network_args):
        print("obsPH: ",observationPH.shape)
        outputC1 = self.conv(observationPH, 8, 8, 4, name = 'Conv1', **network_args)
        print("CNN1: ",outputC1.shape)
        outputC2 = self.conv(outputC1, 16, 4, 2, name = 'Conv2', **network_args)
        print("CNN2: ",outputC2.shape)
        outputFlatten = tf.layers.flatten(outputC2)
        print("Flatten: ",outputFlatten.shape)
        outputFC = self.fc(outputFlatten, 128, name= 'FC1', **network_args)
        print("networkOutput: ",outputFC.shape)
        return outputFC

        ## building cnnlarge consisting of 3 deeper convolutional layers and 1 large FC layer
    def cnnLarge(self, observationPH, **network_args):
        print("obsPH: ",observationPH.shape)
        outputC1 = self.conv(observationPH, 32, 8, 4, name = 'Conv1', **network_args)
        print("CNN1: ",outputC1.shape)
        outputC2 = self.conv(outputC1, 64, 4, 2, name = 'Conv2', **network_args)
        print("CNN2: ",outputC2.shape)
        outputC3 = self.conv(outputC1, 64, 3, 1, name = 'Conv3', **network_args)
        print("CNN3: ",outputC3.shape)
        outputFlatten = tf.layers.flatten(outputC3)
        print("Flatten: ", outputFlatten.shape)
        outputFC = self.fc(outputFlatten, 512, name= 'FC', **network_args)
        print("networkOutput: ",outputFC.shape)
        return outputFC

        ##building Fully connected network with amount of layers and nodes as given by network_args.
        ## 2 layers and 64 units corresponds to the test that were completed in the PPO paper
    def fcNetwork(self, observationPH, numLayers = 2, numNodes=64, **network_args):
        print("obsPH: ",observationPH.shape)
        vector = observationPH
        for i in range(numLayers):
            vector = self.fc(vector, numNodes, **network_args)
            print("layer"+str(i)+": ",vector.shape)
        return vector

    def buildNetwork(self,observationPH,**network_args):  ## select which network to build
        if self.networkOption == 'small':
            print('Small network selected')
            self.networkOutput=self.cnnSmall(observationPH,**network_args)
        elif self.networkOption == 'large':
            print('Large network selected')
            self.networkOutput=self.cnnLarge(observationPH,**network_args)
        elif self.networkOption == 'fc':
            self.networkOutput=self.fcNetwork(observationPH, **network_args)
        else:
            raise ValueError('Invalid network option')

    def createStep(self, **network_args):  ## depending on actor or critic mode, provide the appropiate step output
        if tf.get_variable_scope().name=='actor': ## actor determines the action and the probabilty of that action
            self.outputShape = self.env.action_space.n
            old = network_args['activation']
            network_args['activation'] = None
            self.actionOutput = self.fc(self.networkOutput,self.outputShape, **network_args)
            print("actionspace output: ", self.actionOutput.shape)
            randomizer = tf.random_uniform(tf.shape(self.actionOutput), dtype=tf.float32) # used to explore the environment
            print("randomizer: ", randomizer.shape)
            # self.action = tf.argmax(self.actionOutput- tf.log(-tf.log(randomizer)), axis=1)
            # self.action = tf.argmax(tf.nn.softmax(self.actionOutput), axis=1)
            self.action = tf.multinomial(self.actionOutput,1)
            print("action shape: ", self.action.shape)
            self.logProb = self.logP(self.action)
            print("logprob shape: ", self.logProb.shape)
            network_args['activation'] = old

        elif tf.get_variable_scope().name=='critic': ## critic determines the value of the current state
            old = network_args['activation']
            network_args['activation'] = None
            self.value = self.fc(self.networkOutput, 1, name='Value', **network_args)
            print("Value shape: ", self.value.shape)
            network_args['activation'] = old
        else:
            raise ValueError('no scope detected')

    def logP(self, action): ## function needed to calculate the probabilty of a selected action, both used during stepping trough the environment and during training.
        one_hot_actions = tf.one_hot(action,self.outputShape)
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actionOutput,labels=one_hot_actions)


class agent:
    """
    Provides an agent model that is used to create networks and determining actions based on the observations
    Also provides training functionality for the networks
    env: gym environments
    sess: tensorflow Session
    networkoption: which type of network to created, for this research only fc is used
    epsilon: clipping factor 0.2 is used in the article
    epochs: number of training epochs
    nMinibatch: amount of minibatches that are created from each training set
    loadPath: can be used to load a previously created tensorflow model
    networkStyle: 'copy' uses 2 different networks, 'shared' uses the same network but only uses different output layers for action and value network
    c1: weight of the value function in the total loss function

    """

    def __init__(self, env, sess, networkOption='fc',epsilon = 0.2, epochs = 4, nMiniBatch = 2, loadPath = None,
                    networkStyle = 'copy', c1=0.5, **network_args):
        self.env = env
        self.sess = sess
        self.networkOption = networkOption
        self.epsilon = epsilon
        self.epoch = epochs
        self.nMiniBatch = nMiniBatch
        self.loadPath  = loadPath
        self.shp = list(self.env.observation_space.shape)

        ## Create placeholders used to feed data
        self.observationPH = tf.placeholder(tf.float32,shape=[None]+self.shp, name = "Observation")#,self.shp[1],self.shp[2]]
        self.actionsPH = tf.placeholder(tf.int32,shape=[None],name='Actions')
        self.actionsProbOldPH = tf.placeholder(tf.float32,shape=[None],name='ActionProbOld')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None],name='Advantage')
        self.disRewardsPH = tf.placeholder(tf.float32, shape = [None], name = 'DiscountedRewards')
        self.oldValuePredPH = tf.placeholder(tf.float32, shape = [None], name = 'oldValuePred')
        self.learningRatePH = tf.placeholder(tf.float32, shape = [], name = 'LearningRate')

        if networkStyle == 'copy': ## build netwok using the network class, the scopes what network is created
            with tf.variable_scope('actor'):
                network_args['trainable']=True
                old = network_args.copy()
                print(network_args)
                self.actor = network(self.env, self.networkOption, self.sess)
                self.actor.buildNetwork(self.observationPH,**network_args)
                del network_args['numNodes']
                del network_args['numLayers']
                self.actor.createStep(**network_args)
                self.action = self.actor.action
                self.logProb = self.actor.logProb
                self.logP = self.actor.logP
                print(network_args)
                network_args = old
                print(network_args)

            with tf.variable_scope('critic'):
                network_args['trainable']=True
                print(network_args)
                self.critic = network(self.env, self.networkOption, self.sess)
                self.critic.buildNetwork(self.observationPH,**network_args)
                del network_args['numNodes']
                del network_args['numLayers']
                self.critic.createStep(**network_args)
                self.value = self.critic.value

        elif networkStyle == 'shared':
            with tf.variable_scope('actor'):
                network_args['trainable']=True
                self.shared = network(self.env, self.networkOption, self.sess)
                self.shared.buildNetwork(self.observationPH,**network_args)
                self.shared.createStep(**network_args)
                self.action = self.shared.action
                self.logProb = self.shared.logProb
                self.logP = self.shared.logP

            with tf.variable_scope('critic'):
                network_args['trainable']=True
                self.shared.createStep(**network_args)
                self.value = self.shared.value

        else:
            raise ValueError('networkStyle not recognized')

        with tf.variable_scope('lossFunction'): ## create loss function according to article
            actionsProbNew = self.logP(self.actionsPH) #determine prob of each action given the current network. The old probabilities were already found during the stepping phase
            print("NewProb: ", actionsProbNew.shape)
            ratio = tf.exp(self.actionsProbOldPH - actionsProbNew)  # determine ratio (newProb/oldProb)#negative logprop is given by tensorflow thats why they are switched.. Took me a while to find this out
            print("ratio: ", ratio.shape)
            policyLoss= ratio * self.advantagePH # calculate unclipped policy loss
            print("Policylos: ", policyLoss.shape)
            clippedPolicyLoss= self.advantagePH * tf.clip_by_value(ratio,(1-self.epsilon),(1+self.epsilon)) # calculate clipped policy loss
            print("clippedPolicyLoss: ", clippedPolicyLoss.shape)
            min = tf.minimum(policyLoss, clippedPolicyLoss) # find the minimum of the clipped and unclipped loss
            print("minShape: ", min.shape)
            self.pLoss = -tf.reduce_mean(min) # calculate the average of the entire batch
            print("pLoss: ", self.pLoss.shape)

            value = self.value # calculate value using current network. Old values are already determined in the stepping phase
            print("value: ", self.value.shape)
            valueLoss = tf.square(value - self.disRewardsPH) # calculate squared error value loss
            print("valueLoss: ", valueLoss.shape)
            self.vLoss = tf.reduce_mean(valueLoss) # calculate average valueloss over entire batch
            print("vLoss: ", self.vLoss.shape)
            self.loss = self.pLoss + c1 * self.vLoss # total loss function

            print("loss: ",self.loss.shape)

        with tf.variable_scope('trainer'): ## create trainer using adam and a learning rate as given
            self.train = tf.train.AdamOptimizer(learning_rate= self.learningRatePH).minimize(self.loss)

        self.saver = tf.train.Saver() ## create saver that allows to save and restore the model
        print('Agent created with following properties: ', self.__dict__, network_args)

        if loadPath: ## load model is load path is given
            self.saver.restore(self.sess, self.loadPath)
            print("Model loaded from ", self.loadPath)
        else:
            self.sess.run(tf.global_variables_initializer())

    def step(self, observation):
        # function that uses the observation to calculate the action, logProbabilty and value of the network and returns them
        action, logProb, value = self.sess.run([self.action,self.logProb, self.value], feed_dict= {self.observationPH : observation})
        return  action.squeeze(), logProb.squeeze(), value.squeeze()


    def trainNetwork(self, observations, actions, disRewards, values, actionProbOld, advantage,lr):
        """
        Functions that first creates shuffled minibatches of the data.
        Next it trains the network on those batches for a given amount of epochs
        returns the latest policy and value loss
        """
        lenght = observations.shape[0]
        step = int(lenght/self.nMiniBatch)
        assert(self.nMiniBatch*step == lenght)
        indices = range(0,lenght,step)
        randomIndex = np.arange(lenght)
        for _ in range(self.epoch):
            np.random.shuffle(randomIndex)
            for start in indices:
                end = start+step
                ind = randomIndex[start:end].astype(np.int32)
                observationsB = observations[ind]
                actionsB = actions[ind]
                actionProbOldB = actionProbOld[ind]
                advantageB = advantage[ind]
                disRewardsB = disRewards[ind]
                valuesB = values[ind]
                advantageB = (advantageB - advantageB.mean()) / (advantageB.std() + 1e-8) # normalize the advantage function for faster convergence
                feedDict = {self.observationPH: observationsB, self.oldValuePredPH:valuesB, self.actionsPH: actionsB, self.actionsProbOldPH: actionProbOldB, self.advantagePH: advantageB, self.disRewardsPH: disRewardsB, self.learningRatePH: lr}
                ## feedDict is used to feed the training data to the placeholders created when the agent is initialized
                pLoss, vLoss,  _ = self.sess.run([self.pLoss, self.vLoss, self.train], feedDict)

        return pLoss, vLoss

    def getValue(self, observation): ## functions used to get the value of the current observations. Needed for the advantage calculation
        return (self.sess.run(self.value,{self.observationPH: observation}))

    def saveNetwork(self,name): ## function that saves the current network parameters
        savePath = self.saver.save(self.sess,name)
        print("Model saved in path: %s" % savePath)



## other definitions

def advantageEST(rewards, values, dones, lastValue, gamma, lamda):

    ## using advantage estimator from article
    advantage = np.zeros_like(rewards).astype(np.float32)
    advantage[-1] = lastValue*gamma * (1-dones[-1])+rewards[-1]-values[-1] # calculate latest advantage
    lastAdv = advantage[-1]
    for index in reversed(range(len(rewards)-1)):
        ## Doing it in reverse allows for reuse of already calculated variables and is much faster
        delta = rewards[index] + gamma * (1-dones[index])* values[index+1] -values[index]
        # when an environment was done. The last advantage needs to be ignored as a new episoded begins and the rewards are 0 again
        advantage[index] = lastAdv = delta + gamma * (1-dones[index]) * lamda * lastAdv
    return advantage, (advantage+values)  ## advantage+values= discountedReward according to the article
