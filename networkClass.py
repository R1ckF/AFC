import tensorflow as tf
import tensorflow.contrib.slim as slim

class agent:
    """
    Provides a network model that is used for determining actions based on the observations
    Also provides training functionality
    nCNNLayers: the number of CNN layers
    nFCLayers: the number of fully connected layers

    """

    def __init__(self, env, nCNNLayers, nFCLayers):
        self.env = env

        def createNetwork():
            class network:
                def __init__(self):
                    self.env = env
                    self.nCNNLayers = nCNNLayers
                    self.nFCLayers = nFCLayers
                    self.shp = self.env.observation_space.shape
                    self.X = tf.placeholder(tf.float32,shape=[None, self.shp[0],self.shp[1],self.shp[2]], name = "Observation")
                    self.outputShape = self.env.action_space.n
                    self.actions=cnn()


                def conv(self, input, outputDepth, kernelsize, stride, padding="same", activation_fn=tf.nn.relu):
                      return tf.layers.conv2d(
                      inputs=input,
                      filters=outputDepth,
                      kernel_size=[kernelsize, kernelsize],
                      strides=(stride,stride)
                      padding=padding,
                      activation=activation_fn)

                def fc(self, input, numOutputs, activation_fn=tf.nn.relu):
                    tf.contrib.layers.fully_connected(
                        input,
                        numOutputs,
                        activation_fn=activation_fn,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=initializers.xavier_initializer(),
                        weights_regularizer=None,
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        reuse=None,
                        variables_collections=None,
                        outputs_collections=None,
                        trainable=True,
                        scope=None)

                def cnn(self):
                    print(self.X.shape)
                    outputC1 = self.conv(self.X, 8, 8, 4)
                    print(outputC1.shape)
                    outputC2 = self.conv(outputC1)
                    print(outputC2.shape)
                    outputFC = self.fc(outputC2, self.outputShape)
                    print(outputFC)
                    return tf.nn.softmax(outputFC)


                def step(self, observation):
                    with tf.Session() as sess:
                    return sess.run(action, feed_dict= {self.X =observation})
            return network()
        self.network = createNetwork()
        self.step = self.network.step
