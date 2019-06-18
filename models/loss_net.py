import numpy as np
import os
import scipy.io as sio
import tensorflow as tf

file_path = '../data/vgg_files'
file_name = 'imagenet-vgg-verydeep-19.mat'
vgg_weights = sio.loadmat(os.path.join(file_path, file_name))
vgg_layers = vgg_weights['layers']

layer_dict = {}
for layer_data in vgg_layers[0]:
    name = layer_data['name'][0][0][0]
    weights = layer_data['weights'][0][0]
    layer_dict[name] = weights

class Vgg():

    def __init__(self):
        self.build_vgg()
        self.mean_pixels = np.array([123.68, 116.779, 103.939])

    # def assign_input(image):
        # self.input_image = image

    def _get_weights(self, layer_name):
        """
        Loads the weights for the given convolutional layer of vgg.
        """
        weights, bias = layer_dict[layer_name][0][0], layer_dict[layer_name][0][1]
        return weights, bias

    def _conv2d_relu(self, inputs, layer_name): 
        """
        Initializes the conv and relu layer using vgg weights.
        """
        weights, bias = self._get_weights(layer_name)
        weights = tf.constant(np.transpose(weights, [1, 0, 2, 3]))
        bias = tf.constant(bias.reshape(-1))
        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + bias)

    def _pool(self, inputs):
        return tf.nn.max_pool(inputs, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    def build_vgg(self):
        self.input = tf.placeholder(tf.float32)
        self.relu1_1 = self._conv2d_relu(self.input, 'conv1_1')
        self.relu1_2 = self._conv2d_relu(self.relu1_1, 'conv1_2')
        self.pool1 = self._pool(self.relu1_2) 
        self.relu2_1 = self._conv2d_relu(self.pool1, 'conv2_1')
        self.relu2_2 = self._conv2d_relu(self.relu2_1, 'conv2_2')
        self.pool2 = self._pool(self.relu2_2)
        self.relu3_1 = self._conv2d_relu(self.pool2, 'conv3_1')
        self.relu3_2 = self._conv2d_relu(self.relu3_1, 'conv3_2')
        self.relu3_3 = self._conv2d_relu(self.relu3_2, 'conv3_3')
        self.relu3_4 = self._conv2d_relu(self.relu3_3, 'conv3_4')
        self.pool3 = self._pool(self.relu3_4)
        self.relu4_1 = self._conv2d_relu(self.pool3, 'conv4_1')
        self.relu4_2 = self._conv2d_relu(self.relu4_1, 'conv4_2')
        self.relu4_3 = self._conv2d_relu(self.relu4_2, 'conv4_3')

if __name__ == '__main__':
    import utils
    puppy_image = utils.load_image('../images/content/puppy.jpg')
    vgg = Vgg()
    layers = ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu4_3']
    layer_results = [getattr(vgg, layer) for layer in layers]
    with tf.Session() as sess:
        layer_outputs = sess.run(layer_results, feed_dict={vgg.input:puppy_image})

    #visualize some of the activations of the last relu layer in the list
    import matplotlib.pyplot as plt
    relu_outputs = layer_outputs[4][0]
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    for n in range(16):
        i, j = n//4, n % 4
        axes[i, j].imshow(relu_outputs[:, :, n])
    plt.show()
    




