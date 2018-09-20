import tensorflow as tf
import tensorflow.contrib.slim as slim

class agent:
    """
    Provides a network model that is used for determining actions based on the observations
    Also provides training functionality
    nCNNLayers: the number of CNN layers
    nFCLayers: the number of fully connected layers

    """

    def __init__(self, env, sess, nCNNLayers, nFCLayers):
        self.env = env


        def createNetwork(self):
            class network:
                def __init__(self):
                    self.env = env
                    self.nCNNLayers = nCNNLayers
                    self.nFCLayers = nFCLayers
                    self.sess = sess
                    self.shp = self.env.observation_space.shape
                    self.X = tf.placeholder(tf.float32,shape=[None, self.shp[0],self.shp[1],self.shp[2]], name = "Observation")
                    self.outputShape = self.env.action_space.n
                    self.actions=self.cnn()

                def conv(self, input, outputDepth, kernelsize, stride, padding="same", activation_fn=tf.nn.relu, name = 'Conv'):
                      return tf.layers.conv2d(
                      inputs=input,
                      filters=outputDepth,
                      kernel_size=[kernelsize, kernelsize],strides=(stride,stride),padding=padding,activation=activation_fn, name = name)

                def fc(self, input, numOutputs, activation_fn=tf.nn.relu):
                    tf.layers.dense(
                        input,
                        numOutputs,
                        activation=activation_fn, name = 'FC')

                def cnn(self):
                    print(self.X.shape)
                    outputC1 = self.conv(self.X, 8, 8, 4, name = 'Conv1')
                    print(outputC1.shape)
                    outputC2 = self.conv(outputC1, 16, 4, 2, name = 'Conv2')
                    print(outputC2.shape)
                    print(self.outputShape)
                    outputFC = self.fc(outputC2, self.outputShape)
                    print(outputFC)
                    return outputFC


                def step(self, observation):
                    return self.sess.run(self.actions, feed_dict= {self.X : observation.reshape(1, self.shp[0],self.shp[1],self.shp[2])})

            return network()

        self.network = createNetwork(self)
        self.step = self.network.step
