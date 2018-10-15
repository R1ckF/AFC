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

        ## building cnnlarge
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

        ##building Fully connected network
    def mlp(self, observationPH, numLayers = 2, numNodes=64, **network_args):
        print("obsPH: ",observationPH.shape)
        vector = observationPH
        for i in range(numLayers):
            vector = self.fc(vector, numNodes, **network_args)
            print("layer"+str(i)+": ",vector.shape)
        return vector

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
        if tf.get_variable_scope().name=='actor' or tf.get_variable_scope().name=='actorOld':
            if isinstance(self.env.action_space,gym.spaces.Box):
                print("Continous Control")
                oldActivation = network_args['activation']
                network_args['activation'] = tf.nn.tanh
                self.mu = 2 * self.fc(self.networkOutput,1, name='mu',**network_args)
                network_args['activation'] = tf.nn.softplus
                self.std = self.fc(self.networkOutput,1,name='sigma',**network_args)
                network_args['activation'] = old
                self.dist = tf.distributions.Normal(loc = self.mu, scale = self.std)
                self.params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

            elif isinstance(env.action_space, gym.spaces.Discrete):
                print("Discrete control")
                self.outputShape = env.action_space.n
                self.actionOutput = self.fc(self.networkOutput,self.outputShape, activation=None)
                print("actionspace output: ", actionOutput.shape)
                randomizer = tf.random_uniform(tf.shape(self.outputShape), dtype=tf.float32)
                self.action = tf.argmax(self.actionOutput - tf.log(-tf.log(randomizer)), axis=-1)
                print(action.shape)
                self.logProb = self.logP(self.action)

        elif tf.get_variable_scope().name=='critic':
            self.value = self.fc(self.networkOutput, 1, name='Value', activation=None)
        else:
            raise ValueError('no scope detected')

    def logP(self, action):
        one_hot_actions = tf.one_hot(action,self.outputShape)
        return -tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.actionOutput,labels=one_hot_actions)






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
        self.actionsPH = tf.placeholder(tf.float32,shape=[None,1],name='Actions')
        self.actionsProbOldPH = tf.placeholder(tf.float32,shape=[None,1],name='ActionProbOld')
        self.advantagePH = tf.placeholder(tf.float32, shape=[None,1],name='Advantage')
        self.disRewardsPH = tf.placeholder(tf.float32, shape = [None,1], name = 'DiscountedRewards')
        self.oldValuePredPH = tf.placeholder(tf.float32, shape = [None, 1], name = 'oldValuePred')

        if cnnStyle == 'copy':
            with tf.variable_scope('actor'):
                network_args['trainable']=True
                self.actor = network(self.env, self.networkOption, self.sess)
                self.actor.buildNetwork(self.observationPH,**network_args)
                self.actor.createStep(**network_args)
                self.action = self.actor.action
                self.logP = self.actor.logP

            with tf.variable_scope('critic'):
                network_args['trainable']=True
                self.critic = network(self.env, self.networkOption, self.sess)
                self.critic.buildNetwork(self.observationPH,**network_args)
                self.critic.createStep(**network_args)
                self.value = self.critic.value
                self.advantage = self.disRewardsPH - self.value
                self.valueLoss = tf.reduce_mean(tf.square(self.advantage))



        elif cnnStyle == 'shared':
            with tf.variable_scope('actor'):
                network_args['trainable']=True
                self.shared = network(self.env, self.networkOption, self.sess)
                self.shared.buildNetwork(self.observationPH,**network_args)
                self.shared.createStep(**network_args)
                self.action = self.shared.action
                self.logP = self.shared.logP

            with tf.variable_scope('critic'):
                network_args['trainable']=True
                self.shared.createStep(**network_args)
                self.value = self.shared.value

        else:
            raise ValueError('cnnStyle not recognized')

        with tf.variable_scope('lossFunction'):
            actionsProbNew = self.logP(self.actionsPH)
            ratio = tf.exp(actionsProbNew - self.actionsProbOldPH)
            policyLoss= ratio * self.advantagePH
            clippedPolicyLoss= self.advantagePH * tf.clip_by_value(ratio,(1-self.epsilon),(1+self.epsilon))
            self.pLoss = tf.reduce_mean(tf.minimum(policyLoss, clippedPolicyLoss))

            value = self.value
            vpredclipped = self.oldValuePredPH + tf.clip_by_value(value - self.oldValuePredPH , - self.epsilon, self.epsilon)
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)

        with tf.variable_scope('trainer'):


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

        action, mu, sigma, l1 = self.sess.run([self.action, self.actor.mu, self.actor.std, self.actor.networkOutput], feed_dict= {self.observationPH : observation})#, options=run_options, run_metadata=run_metadata)
        # print("sess time: ", time.time()-tt)
        # writer.add_run_metadata(run_metadata, 'step%d' % i)

        return np.clip(action, -2, 2), mu, sigma, l1



    def get_Value(self, obs):
        if obs.ndim < 2: obs = obs[np.newaxis,:]
        return self.sess.run(self.value,{self.observationPH:obs})


    def trainNetwork(self, observations, actions, disRewards):
        self.sess.run(self.updateActorOld)
        adv = self.sess.run(self.advantage, {self.observationPH: observations, self.disRewardsPH: disRewards})
        # adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        # print(adv)
        # for _ in range(self.epoch):
        #     aLoss, _ = self.sess.run([self.aLoss, self.atrain], {self.observationPH: observations, self.actionsPH: actions, self.advantagePH: adv})
        #
        # for _ in range(self.epoch):
        #     cLoss, _  = self.sess.run([self.cLoss, self.ctrain], {self.observationPH: observations, self.disRewardsPH: disRewards})
        #
        # return aLoss, cLoss



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
                # actionProbOldB = actionProbOld[ind]
                advantageB = adv[ind]
                disRewardsB = disRewards[ind]
                aLoss, cLoss,  _ = self.sess.run([self.aLoss, self.cLoss, self.atrain], {self.observationPH: observationsB, self.disRewardsPH: disRewardsB, self.actionsPH: actionsB, self.advantagePH: advantageB})
                # cLoss, _  = self.sess.run([self.cLoss, self.ctrain], {self.observationPH: observationsB, self.disRewardsPH: disRewardsB})

        return aLoss, cLoss
                # feedDict = {self.observationPH: observationsB, self.actionsPH: actionsB, self.actionsProbOldPH: actionProbOldB, self.advantagePH: advantageB, self.disRewardsPH: disRewardsB}
        #         lClip, lVF, entropy, _ ,_ = self.sess.run([self.lCLIP, self.lVF, self.entropy, self.trainA, self.trainC],feed_dict = feedDict)
        # return lClip, lVF, entropy

    def saveNetwork(self,name):
        savePath = self.saver.save(self.sess,name)
        print("Model saved in path: %s" % savePath)

## other definitions

def advantageDR(rewards, gamma, v_s):

    disRewards = []
    for r in rewards[::-1]:
        v_s = r[0]+gamma*v_s
        disRewards.append(v_s)
    disRewards.reverse()
    return np.asarray(disRewards,dtype=np.float32).reshape(-1,1)
    # v_s = 4
    # ## using discounted rewards to get advantage
    # disRewards = []
    # for r in rewards[::-1]:
    #     v_s = r+gamma*v_s
    #     disRewards.append(v_s)
    # return disRewards.reverse()
    # disRewards = np.asarray(rewards,dtype=np.float32)
    # disRewards[0] = values[-1]
    # for index in range(1,len(rewards)):
    #     disRewards[index] = disRewards[index-1]*gamma + rewards[index]
    # return disRewards[::-1] - values, disRewards[::-1]

def advantageEST(rewards, values, gamma, lamda):

    ## using advantage estimator from article
    advantage = np.zeros_like(rewards)
    advantage[-1] = values[-1]*gamma+rewards[-1]-values[-1]
    lastAdv = advantage[-1]
    for index in reversed(range(len(rewards)-1)):
        delta = rewards[index] + gamma * values[index+1] -values[index]
        advantage[index] = lastAdv = delta + gamma * lamda * lastAdv
    return advantage.reshape((-1,1)), (advantage+values).reshape((-1,1))

# print(advantageDR(np.asarray([2,5,4,5,2,3,1,6],dtype=np.float32).reshape((-1,1)),0.9,4))
# print(advantageEST([2,5,4,5,2,3,1,6],[4,4,4,4,4,4,4,4],0.9,1))
