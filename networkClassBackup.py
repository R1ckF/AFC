import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time
import gym

class network:
    """
    This class is used as a framework that can be called to create actor or critic networks.
    networkOption: the network that is selected
        large: consists of ...
        small: consists of ...
    """
    def __init__(self,env, networkOption, sess):
        self.env = env
        self.networkOption = networkOption
        # self.outputShape = self.env.action_space.n

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
        outputFC = self.fc(outputFlatten, 512, name= 'FC', **network_args)
        print(outputFC.shape)
        return outputFC

    def mlp(self, observationPH, numNodes=100, **network_args):
        print(observationPH.shape)
        outputFC1 = self.fc(observationPH, numNodes, name = 'FC1', **network_args)
        print(outputFC1.shape)
        # outputFC2 = self.fc(outputFC1, numNodes, name = 'FC2', **network_args)
        # print(outputFC2.shape)
        return outputFC1

    def buildNetwork(self,observationPH,**network_args):
        if self.networkOption == 'small':
            print('Small network selected')
            self.networkOutput=self.cnnSmall(observationPH,**network_args)
        elif self.networkOption == 'large':
            print('Large network selected')
            self.networkOutput=self.cnnLarge(observationPH,**network_args)
        elif self.networkOption == 'mlp':
            self.networkOutput=self.mlp(observationPH, **network_args)
        else:
            raise ValueError('Invalid network option')

    def createStep(self, **network_args):
        if tf.get_variable_scope().name=='actor':
            if isinstance(self.env.action_space,gym.spaces.Box):
                print("Continous Control")
                mu = 2 * self.fc(self.networkOutput,1, name='mu',activation= tf.nn.tanh)
                std = self.fc(self.networkOutput,1,name='sigma',activation=tf.nn.softplus)
                self.dist = tf.distributions.Normal(loc = mu, scale = std)
            elif isinstance(env.action_space, gym.spaces.Discrete):
                outputShape = env.action_space.n


            # self.action = tf.nn.softmax(self.fc(self.networkOutput, self.outputShape, name= 'Action', **network_args))
            # self.action = self.dist.sample(1)
        elif tf.get_variable_scope().name=='critic':
            self.value = self.fc(self.networkOutput, 1, name='Value', activation=None)
        else:
            raise ValueError('no scope detected')


    def lossFunction(self,actionsPH, actionsProbOldPH, advantagePH, disRewardsPH, e):
        if tf.get_variable_scope().name=='actor':
            actionLogProbNew = self.dist.log_prob(actionsPH)
            # actionLogProbNew = tf.Print(actionLogProbNew2, [actionLogProbNew2, actionsProbOldPH])
            ratio = tf.exp(actionLogProbNew - actionsProbOldPH)
            # ratio = tf.Print(ratio2,[(ratio2)])
            Lcpi = ratio * advantagePH
            clipped = tf.clip_by_value(ratio,(1-e),(1+e))*advantagePH
            self.lCLIP = tf.reduce_mean(tf.minimum(Lcpi,clipped))
            self.meanEntropy = tf.reduce_mean(self.dist.entropy())
            return self.lCLIP, self.meanEntropy
        elif tf.get_variable_scope().name=='critic':
            self.lVF = tf.square(self.value-disRewardsPH)
            self.meanlVF = tf.reduce_mean(self.lVF)
            return self.meanlVF
        else:
            raise ValueError('no scope detected')



class agent:
    """
    Provides an agent model that is used for to create networks and determining actions based on the observations
    Also provides training functionality for the networks
    """

    def __init__(self, env, sess, networkOption='small',epsilon = 0.2, epochs = 3, learningRate = 0.0005, nMiniBatch = 2, loadPath = None,
                    cnnStyle = 'copy', c1=1, c2 = 0.01, **network_args):
        self.env = env
        self.sess = sess
        self.networkOption = networkOption
        self.epsilon = epsilon
        self.epoch = epochs
        self.nMiniBatch = nMiniBatch
        self.loadPath  = loadPath
        self.shp = self.env.observation_space.shape
        self.observationPH = tf.placeholder(tf.float32,shape=[None, self.shp[0]], name = "Observation")#,self.shp[1],self.shp[2]]
        # self.outputShape = self.env.action_space.n
        self.actionsPH = tf.placeholder(tf.float32,shape=[None,1],name='Actions')
        self.actionsProbOldPH = tf.placeholder(tf.float32,shape=[None,1],name='ActionProbOld')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None,1],name='Advantage')
        self.disRewardsPH = tf.placeholder(tf.float32, shape = [None,1], name = 'DiscountedRewards')

        if cnnStyle == 'copy':
            with tf.variable_scope('actor'):
                self.actor = network(self.env, self.networkOption, self.sess)
                self.actor.buildNetwork(self.observationPH,**network_args)
                self.actor.createStep(**network_args)
                self.lCLIP, self.entropy = self.actor.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)

            with tf.variable_scope('critic'):
                self.critic = network(self.env, self.networkOption, self.sess)
                self.critic.buildNetwork(self.observationPH,**network_args)
                self.critic.createStep(**network_args)
                self.lVF = self.critic.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)
            self.dist = self.actor.dist
            self.action = self.dist.sample(1)
            self.value = self.critic.value
            self.logProb = self.dist.log_prob(self.action)

        elif cnnStyle == 'shared':
            with tf.variable_scope('sharedCNN'):
                self.shared = network(self.env, self.networkOption, self.sess)
                self.shared.buildNetwork(self.observationPH, **network_args)
            with tf.variable_scope('actor'):
                self.shared.createStep(**network_args)
                self.lCLIP, self.entropy = self.shared.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)
            with tf.variable_scope('critic'):
                self.shared.createStep(**network_args)
                self.lVF = self.shared.lossFunction(self.actionsPH, self.actionsProbOldPH, self.advantagePH, self.disRewardsPH, self.epsilon)
            self.dist = self.shared.dist
            self.action = self.dist.sample(1)
            self.value = self.shared.value
            self.logProb = self.dist.log_prob(self.action)

        else:
            raise ValueError('cnnStyle not recognized')

        with tf.variable_scope('trainer'):
            self.lossFunction = -(self.lCLIP - c1 * self.lVF + c2 * self.entropy)
            self.Aloss = -self.lCLIP
            self.Closs = self.lVF
            #self.printList = [ self.lCLIP, tf.shape(self.shared.entropy),tf.shape(self.shared.value),tf.shape(self.shared.lVF)]
            #self.lossPrint = tf.Print(self.lossFunction,self.printList)
            self.trainA = tf.train.AdamOptimizer(learning_rate= learningRate).minimize(self.Aloss)
            self.trainC = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(self.Closs)

        self.saver = tf.train.Saver()
        print('Agent created with following properties: ', self.__dict__, network_args)

        if loadPath:
            self.saver.restore(self.sess, self.loadPath)
            print("Model loaded from ", self.loadPath)
        else:
            self.sess.run(tf.global_variables_initializer())

    def step(self, observation,writer,i):

        # self.action = tf.Print(self.action2, [self.action2, self.dist.loc,self.dist.scale])

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # tt = time.time()
        action , logProb, value = self.sess.run([self.action, self.logProb, self.value], feed_dict= {self.observationPH : observation})#, options=run_options, run_metadata=run_metadata)
        # print("sess time: ", time.time()-tt)
        # writer.add_run_metadata(run_metadata, 'step%d' % i)

        return np.clip(action.squeeze(), -2, 2).reshape(-1,1), value.squeeze(), logProb.squeeze()

    def trainNetwork(self, observations, actions, actionProbOld, advantage, disRewards):
        l = observations.shape[0]
        step = int(l/self.nMiniBatch)
        assert(self.nMiniBatch*step == l)
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
                feedDict = {self.observationPH: observationsB, self.actionsPH: actionsB, self.actionsProbOldPH: actionProbOldB, self.advantagePH: advantageB, self.disRewardsPH: disRewardsB}
                lClip, lVF, entropy, _ ,_ = self.sess.run([self.lCLIP, self.lVF, self.entropy, self.trainA, self.trainC],feed_dict = feedDict)
        return lClip, lVF, entropy

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
