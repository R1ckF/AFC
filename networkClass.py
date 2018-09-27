import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class network:
    """
    This class is used as a framework that can be called to create actor or critic networks.
    CNNoption: the network that is selected
        large: consists of ...
        small: consists of ...
    """
    def __init__(self,env, CNNoption, sess):
        self.env = env
        self.CNNoption = CNNoption
        self.outputShape = self.env.action_space.n

        ## function that creates a convolution layers
    def conv(self, input, outputDepth, kernelsize, stride, padding="valid", name = 'Conv', **network_args):
        return tf.layers.conv2d(
          inputs=input,
          filters=outputDepth,
          kernel_size=[kernelsize, kernelsize],strides=(stride,stride),padding=padding, name = name, **network_args)

          ##Function that creates a fully connected layer
    def fc(self, input, numOutputs, name = 'FC', **network_args):
        return tf.layers.dense(
            input,
            numOutputs,
            name = name, **network_args)

        ## building cnnsmall
    def cnnSmall(self, observationPH, **network_args):
        print(observationPH.shape)
        outputC1 = self.conv(observationPH, 8, 8, 4, name = 'Conv1', **network_args)
        print(outputC1.shape)
        outputC2 = self.conv(outputC1, 16, 4, 2, name = 'Conv2', **network_args)
        print(outputC2.shape)
        outputFlatten = tf.layers.flatten(outputC2)
        print(outputFlatten.shape)
        print(self.outputShape)
        outputFC = self.fc(outputFlatten, 128, name= 'FC1', **network_args)
        print(outputFC.shape)
        return outputFC

        ## building cnnlarge
    def cnnLarge(self, observationPH, **network_args):
        print(observationPH.shape)
        outputC1 = self.conv(observationPH, 32, 8, 4, name = 'Conv1', **network_args)
        print(outputC1.shape)
        outputC2 = self.conv(outputC1, 64, 4, 2, name = 'Conv2', **network_args)
        print(outputC2.shape)
        outputC3 = self.conv(outputC1, 64, 3, 1, name = 'Conv3', **network_args)
        outputFlatten = tf.layers.flatten(outputC3)
        print(outputFlatten.shape)
        print(self.outputShape)
        outputFC = self.fc(outputFlatten, 512, name= 'FC', **network_args)
        print(outputFC.shape)
        return outputFC

    def buildNetwork(self,observationPH,**network_args):
        if self.CNNoption == 'small':
            print('Small network selected')
            self.networkOutput=self.cnnSmall(observationPH,**network_args)
        elif self.CNNoption == 'large':
            print('Large network selected')
            self.networkOutput=self.cnnLarge(observationPH,**network_args)
        else:
            raise ValueError('Invalid network option')

    def createStep(self, **network_args):
        if tf.get_variable_scope().name=='actor':
            self.action = tf.nn.softmax(self.fc(self.networkOutput, self.outputShape, name= 'Action', **network_args))
        elif tf.get_variable_scope().name=='critic':
            self.value = self.fc(self.networkOutput, 1, name='Value', **network_args)
        else:
            raise ValueError('no scope detected')


    def lossFunction(self,actionsPH, actionsProbOldPH, advantagePH, disRewardsPH, e):
        if tf.get_variable_scope().name=='actor':
            ratio = tf.gather_nd(self.action, actionsPH) / actionsProbOldPH
            Lcpi = ratio * advantagePH
            clipped = tf.clip_by_value(ratio,(1-e),(1+e))*advantagePH
            self.lCLIP = tf.reduce_mean(tf.minimum(Lcpi,clipped))
            self.entropy = tf.reduce_mean((tf.reduce_sum(-self.action*tf.log(self.action),axis=1)))
            return self.lCLIP, self.entropy
        elif tf.get_variable_scope().name=='critic':
            self.lVF = tf.reduce_mean(tf.square(self.value-disRewardsPH))
            return self.lVF
        else:
            raise ValueError('no scope detected')



class agent:
    """
    Provides an agent model that is used for to create networks and determining actions based on the observations
    Also provides training functionality for the networks
    """

    def __init__(self, env, sess, CNNoption='small',epsilon = 0.2, epochs = 10, learningRate = 0.0005, nMiniBatch = 2, loadPath = None,
                    cnnStyle = 'copy', c1=1, c2 = 0.01, **network_args):
        self.env = env
        self.sess = sess
        self.CNNoption = CNNoption
        self.epsilon = epsilon
        self.epoch = epochs
        self.nMiniBatch = nMiniBatch
        self.loadPath  = loadPath
        self.shp = self.env.observation_space.shape
        self.observationPH = tf.placeholder(tf.float32,shape=[None, self.shp[0],self.shp[1],self.shp[2]], name = "Observation")
        self.outputShape = self.env.action_space.n
        self.actionsPH = tf.placeholder(tf.int32,shape=[None,2],name='Actions')
        self.actionsProbOldPH = tf.placeholder(tf.float32,shape=[None,1],name='ActionProbOld')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None,1],name='Advantage')
        self.disRewardsPH = tf.placeholder(tf.float32, shape = [None,1], name = 'DiscountedRewards')

        if cnnStyle == 'copy':
            with tf.variable_scope('actor'):
                self.actor = network(self.env, self.CNNoption, self.sess)
                self.actor.buildNetwork(self.observationPH,**network_args)
                self.actor.createStep(**network_args)
                self.lCLIP, self.entropy = self.actor.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)

            with tf.variable_scope('critic'):
                self.critic = network(self.env, self.CNNoption, self.sess)
                self.critic.buildNetwork(self.observationPH,**network_args)
                self.critic.createStep(**network_args)
                self.lVF = self.critic.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)
            self.action = self.actor.action
            self.value = self.critic.value

        elif cnnStyle == 'shared':
            with tf.variable_scope('sharedCNN'):
                self.shared = network(self.env, self.CNNoption, self.sess)
                self.shared.buildNetwork(self.observationPH, **network_args)
            with tf.variable_scope('actor'):
                self.shared.createStep(**network_args)
                self.lCLIP, self.entropy = self.shared.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)
            with tf.variable_scope('critic'):
                self.shared.createStep(**network_args)
                self.lVF = self.shared.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)
            self.action = self.shared.action
            self.value = self.shared.value

        else:
            raise ValueError('cnnStyle not recognized')

        with tf.variable_scope('trainer'):
            self.lossFunction = -(self.lCLIP - c1 * self.lVF + c2 * self.entropy)
            self.train = tf.train.AdamOptimizer(learning_rate= learningRate).minimize(self.lossFunction)

        self.saver = tf.train.Saver()
        print('Agent created with following properties: ', self.__dict__, network_args)

        if loadPath:
            self.saver.restore(self.sess, self.loadPath)
            print("Model loaded from ", self.loadPath)
        else:
            self.sess.run(tf.global_variables_initializer())

    def step(self, observation):
        actionSpace, value = self.sess.run([self.action, self.value], feed_dict= {self.observationPH : observation})
        action = np.random.choice(self.outputShape,p=actionSpace.squeeze())  ## selecting action with probabilities according to softmax layer
        return action, value.squeeze(), actionSpace.squeeze()[action]

    def trainNetwork(self, observations, actions, actionProbOld, advantage, disRewards):
        l = observations.shape[0]
        step = int(l/self.nMiniBatch)
        indices = range(0,l,step)
        randomIndex = np.arange(l)
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
                actionsB = np.transpose(np.vstack((np.arange(len(actionsB),dtype=np.int32),np.asarray(actionsB,dtype=np.int32))))
                feedDict = {self.observationPH: observationsB, self.actionsPH: actionsB, self.actionsProbOldPH: actionProbOldB, self.advantagePH: advantageB, self.disRewardsPH: disRewardsB}
                self.sess.run(self.train,feed_dict = feedDict)

    def saveNetwork(self,name):
        savePath = self.saver.save(self.sess,name)
        print("Model saved in path: %s" % savePath)

## other definitions
def advantageDR(rewards, values, gamma):

    ## using discounted rewards to get advantage
    disRewards = np.asarray(rewards,dtype=np.float32)
    disRewards[0] = values[-1]
    for index in range(1,len(rewards)):
        disRewards[index] = disRewards[index-1]*gamma + rewards[index]
    return disRewards[::-1] - values, disRewards[::-1]

def advantageEST(rewards, values, gamma, lamda):

    ## using advantage estimator from article
    advantage = np.zeros_like(rewards)
    advantage[-1] = values[-1]*gamma+rewards[-1]-values[-1]
    lastAdv = advantage[-1]
    for index in reversed(range(len(rewards)-1)):
        delta = rewards[index] + gamma * values[index+1] -values[index]
        advantage[index] = lastAdv = delta + gamma * lamda * lastAdv
    return advantage.reshape((-1,1)), (advantage+values).reshape((-1,1))
