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
    def cnnSmall(self, X, **network_args):
        print(X.shape)
        outputC1 = self.conv(X, 8, 8, 4, name = 'Conv1', **network_args)
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
    def cnnLarge(self, X, **network_args):
        print(X.shape)
        outputC1 = self.conv(X, 32, 8, 4, name = 'Conv1', **network_args)
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

    def buildNetwork(self,X,**network_args):
        if self.CNNoption == 'small':
            print('Small network selected')
            self.networkOutput=self.cnnSmall(X,**network_args)
        elif self.CNNoption == 'large':
            print('Large network selected')
            self.networkOutput=self.cnnLarge(X,**network_args)
        else:
            raise ValueError('Invalid network option')

        if tf.get_variable_scope().name=='actor':
            self.action = tf.nn.softmax(self.fc(self.networkOutput, self.outputShape, name= 'Action', **network_args))
        elif tf.get_variable_scope().name=='critic':
            self.value = self.fc(self.networkOutput, 1, name='Value', **network_args)

    if tf.get_variable_scope().name=='actor':
        def updateNetwork(self,actionsProbOldPH, advantagePH):
            ratio = tf.gather_nd(self.action, actionsPH) / actionsProbOldPH
            Lcpi = ratio * advantagePH
            clipped = tf.clip_by_value(ratio,(1-e),(1+e))*advantagePH
            self.lossFunction = -tf.reduce_mean(tf.minimum(Lcpi,clipped))

    elif tf.get_variable_scope().name=='critic':
        def updateNetwork(self,disRewardsPH):
            self.lossFunction = tf.reduce_mean(tf.square(self.critic.value-disRewardsPH))



class agent:
    """
    Provides an agent model that is used for to create networks and determining actions based on the observations
    Also provides training functionality for the networks
    """

    def __init__(self, env, sess, CNNoption='small', **network_args):
        self.env = env
        self.sess = sess
        self.CNNoption = CNNoption
        self.sess = sess
        self.shp = self.env.observation_space.shape
        self.X = tf.placeholder(tf.float32,shape=[None, self.shp[0],self.shp[1],self.shp[2]], name = "Observation")
        self.outputShape = self.env.action_space.n
        self.actionsPH = tf.placeholder(tf.float32,shape=[None,2],name='Actions')
        self.actionsProbOldPH = tf.placeholder(tf.float32,shape=[None,1],name='ActionProbOld')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None,1],name='Advantage')
        self.disRewardsPH = tf.placeholder(tf.float32, shape = [None,1], name = 'Discounted Rewards')

        with tf.variable_scope('actor'):
            network_args['trainable'] = True
            self.actor = network(self.env, self.CNNoption, self.sess)
            self.actor.buildNetwork(self.X,**network_args)
        # with tf.variable_scope('actorOld'):
        #     network_args['trainable'] = False
        #     self.actorOld = network(self.env, self.CNNoption, self.sess)
        #     self.actorOld.buildNetwork(self.X,**network_args)
        with tf.variable_scope('critic'):
            network_args['trainable'] = True
            self.critic = network(self.env, self.CNNoption, self.sess)
            self.critic.buildNetwork(self.X,**network_args)
        self.sess.run(tf.global_variables_initializer())

    def step(self, observation):
        actionSpace, value = self.sess.run([self.actor.action, self.critic.value], feed_dict= {self.X : observation})
        print(actionSpace)
        action = np.random.choice(self.outputShape,1,p=actionSpace.squeeze())
        return action, value, actionSpace.squeeze()[action]

    def getValue(self, observation):
        self.sess.run(self.critic.value, feed_dict={self.X: observation})

    def updateNetwork(self,action, ActionSpaceProb,Values,DisRewards,e):


        self.lossFunctionCritic = tf.reduce_mean(tf.square(self.critic.value-disRewardsPH))





## other definitions
def advantageDR(rewards, values, gamma):

    ## using discounted rewards to get advantage
    disRewards = np.asarray(rewards,dtype=np.float32)
    disRewards[0] = values[-1]
    for index in range(1,len(rewards)):
        disRewards[index] = disRewards[index-1]*gamma + rewards[index]
    return disRewards[::-1] - values, disRewards[::-1]

def advantageEST(rewards, values, gamma):

    ## using advantage estimator from article
    advantage = np.zeros_like(rewards)
    print(advantage)
    advantage[-1] = values[-1]*gamma+rewards[-1]-values[-1]
    lastAdv = advantage[-1]
    for index in reversed(range(len(rewards)-1)):
        delta = rewards[index] + gamma * values[index+1] -values[index]
        advantage[index] = lastAdv = delta + gamma * lastAdv
    return advantage
