import numpy as np
import os
import scipy.io as sio
import tensorflow as tf

import utils

file_path = '../data/vgg_files'
file_name = 'imagenet-vgg-verydeep-19.mat'
vgg_weights = sio.loadmat(os.path.join(file_path, file_name))
vgg_layers = vgg_weights['layers']

layer_dict = {}
for layer_data in vgg_layers[0]:
    name = layer_data['name'][0][0][0]
    weights = layer_data['weights'][0][0]
    layer_dict[name] = weights

MEAN_PIXELS = np.array([123.68, 116.779, 103.939], dtype=np.float32)


def vgg(graph_input):
    """
    Implements a portion of the Vgg network and returns a dictionary
    with the relevant activations used for the calculation of
    the loss function.
    """
    relu1_1 = _conv2d_relu(graph_input, 'conv1_1')
    relu1_2 = _conv2d_relu(relu1_1, 'conv1_2')
    pool1 = _pool(relu1_2, 'pool1')
    relu2_1 = _conv2d_relu(pool1, 'conv2_1')
    relu2_2 = _conv2d_relu(relu2_1, 'conv2_2')
    pool2 = _pool(relu2_2, 'pool2')
    relu3_1 = _conv2d_relu(pool2, 'conv3_1')
    relu3_2 = _conv2d_relu(relu3_1, 'conv3_2')
    relu3_3 = _conv2d_relu(relu3_2, 'conv3_3')
    relu3_4 = _conv2d_relu(relu3_3, 'conv3_4')
    pool3 = _pool(relu3_4, 'pool3')
    relu4_1 = _conv2d_relu(pool3, 'conv4_1')
    relu4_2 = _conv2d_relu(relu4_1, 'conv4_2')
    relu4_3 = _conv2d_relu(relu4_2, 'conv4_3')
    relu4_4 = _conv2d_relu(relu4_3, 'conv4_4')

    activations = {}
    activations['relu1_2'] = relu1_2
    activations['relu2_2'] = relu2_2
    activations['relu3_4'] = relu3_4
    activations['relu4_4'] = relu4_4
    return activations

def normalize(image):
    return image - MEAN_PIXELS

def denormalize(image):
    return image + MEAN_PIXELS

def _get_weights(layer_name):
    """
    Loads the weights for the given convolutional layer of vgg.
    """
    weights, bias = layer_dict[layer_name][0][0], layer_dict[layer_name][0][1]
    return weights, bias

def _conv2d_relu(inputs, layer_name):
    """
    Initializes the conv and relu layer using vgg weights.
    """
    with tf.name_scope('vgg_' + layer_name):
        weights, bias = _get_weights(layer_name)
        weights = tf.constant(np.transpose(weights, [1, 0, 2, 3]))
        bias = tf.constant(bias.reshape(-1))
        conv = tf.nn.conv2d(inputs, weights,
                            strides=[1, 1, 1, 1], padding='SAME')
        return tf.nn.relu(conv + bias)

def _pool(inputs, name):
    with tf.name_scope('vgg_' + name):
        return tf.nn.max_pool(inputs, ksize=(1, 2, 2, 1),
                              strides=(1, 2, 2, 1), padding='SAME')

if __name__ == '__main__':
    """
    Visualize some layers for a puppy image
    """
    puppy_image = utils.load_image('../images/content/puppy.jpg', expand_dims=True)
    layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']
    layer_dict = vgg(puppy_image)
    layer_results = [layer_dict[layer] for layer in layers]
    with tf.Session() as sess:
        layer_outputs = sess.run(layer_results)

    #visualize some of the activations of the last relu layer in the list
    import matplotlib.pyplot as plt
    relu_outputs = layer_outputs[-2][0]
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True)
    for n in range(16):
        i, j = n//4, n % 4
        axes[i, j].imshow(relu_outputs[:, :, n])
    plt.show()





