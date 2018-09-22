import tensorflow as tf
import tensorflow.contrib.slim as slim

class agent:
    """
    Provides a network model that is used for determining actions based on the observations
    Also provides training functionality
    CNNoption: the network that is selected
        large: consists of ...
        small: consists of ...

    """

    def __init__(self, env, sess, CNNoption):
        self.env = env


        def createNetwork(self):
            class network:
                def __init__(self):
                    self.env = env
                    self.CNNoption = CNNoption
                    self.sess = sess
                    self.shp = self.env.observation_space.shape
                    self.X = tf.placeholder(tf.float32,shape=[None, self.shp[0],self.shp[1],self.shp[2]], name = "Observation")
                    self.outputShape = self.env.action_space.n
                    if self.CNNoption == 'small':
                        self.actions=self.cnnSmall()
                    elif self.CNNoption == 'large':
                        self.actions=self.cnnLarge()
                    else:
                        print('Invalid network option')
                    self.sess.run(tf.global_variables_initializer())

                    ## function that creates a convolution layers
                def conv(self, input, outputDepth, kernelsize, stride, padding="valid", activation_fn=tf.nn.relu, name = 'Conv'):
                      return tf.layers.conv2d(
                      inputs=input,
                      filters=outputDepth,
                      kernel_size=[kernelsize, kernelsize],strides=(stride,stride),padding=padding,activation=activation_fn, name = name)

                      ##Function that creates a fully connected layer
                def fc(self, input, numOutputs, activation_fn=tf.nn.relu, name = 'FC'):
                    return tf.layers.dense(
                        input,
                        numOutputs,
                        activation=activation_fn, name = name)

                    ## building cnnsmall
                def cnnSmall(self):
                    print(self.X.shape)
                    outputC1 = self.conv(self.X, 8, 8, 4, name = 'Conv1')
                    print(outputC1.shape)
                    outputC2 = self.conv(outputC1, 16, 4, 2, name = 'Conv2')
                    print(outputC2.shape)
                    outputFlatten = tf.layers.flatten(outputC2)
                    print(outputFlatten.shape)
                    print(self.outputShape)
                    outputFC = self.fc(outputFlatten, self.outputShape, name= 'FC')
                    print(outputFC.shape)
                    return outputFC

                    ## building cnnlarge
                def cnnLarge(self):
                    print(self.X.shape)
                    outputC1 = self.conv(self.X, 32, 8, 4, name = 'Conv1')
                    print(outputC1.shape)
                    outputC2 = self.conv(outputC1, 64, 4, 2, name = 'Conv2')
                    print(outputC2.shape)
                    outputC3 = self.conv(outputC1, 64, 3, 1, name = 'Conv3')
                    outputFlatten = tf.layers.flatten(outputC3)
                    print(outputFlatten.shape)
                    print(self.outputShape)
                    outputFC = self.fc(outputFlatten, self.outputShape, name= 'FC')
                    print(outputFC.shape)
                    return outputFC

                    ##step function that can be called to get an action space output from the actor network
                def step(self, observation):
                    return self.sess.run(self.actions, feed_dict= {self.X : observation})

            return network()

        self.network = createNetwork(self)
        self.step = self.network.step
