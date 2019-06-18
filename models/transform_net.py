import numpy as np
import tensorflow as tf

INIT_STDDEV = 0.1

def transform_net(inputs):
    conv_1 = _conv_layer(inputs, 32, 9, 1)
    conv_2 = _conv_layer(conv_1, 64, 3, 2)
    conv_3 = _conv_layer(conv_2, 128, 3, 2)
    res_block_1 = _residual_block(conv_3)
    res_block_2 = _residual_block(res_block_1)
    res_block_3 = _residual_block(res_block_2)
    res_block_4 = _residual_block(res_block_3)
    res_block_5 = _residual_block(res_block_4)
    conv_4 = _conv_layer(res_block_5, 64, 3, 2, transpose=True)
    conv_5 = _conv_layer(conv_4, 32, 3, 2, transpose=True)
    output = _conv_layer(conv_5, 3, 9, 1, transpose=True, use_relu=False)
    scaled_output = tf.nn.tanh(output) * 255/2 + 255/2
    return output

def _init_kernel(inputs, n_filters, shape, transpose=False):
    in_channels = tf.shape(inputs)[-1]
    if transpose:
        kernel_shape = [shape, shape, n_filters, in_channels]
    else:
        kernel_shape = [shape, shape, in_channels, n_filters]
    kernel_init = tf.truncated_normal(kernel_shape, stddev=INIT_STDDEV)
    return tf.Variable(kernel_init, dtype=tf.float32)

def _conv_layer(inputs, num_filters, filter_shape, stride, transpose=False, use_relu=True):
    """
    Makes a convolutional layer that pads to keep the input the 
    same size as the output and applies instance normalization
    on the activations.
    """
    kernel = _init_kernel(inputs, num_filters, filter_shape, transpose)
    if transpose:
        batch, height, width, in_channels = inputs.get_shape()
        output_shape = tf.stack([batch, int(stride * height), int(stride * width), num_filters])
        conv_output = tf.nn.conv2d_transpose(inputs, kernel, output_shape, [1, stride, stride, 1], padding='SAME')
    else:
        conv_output = tf.nn.conv2d(inputs, filter=kernel, strides=[1, stride, stride, 1], padding='SAME')
    bias_init = tf.zeros([num_filters])
    bias = tf.Variable(bias_init, dtype=tf.float32)
    conv_output = tf.nn.bias_add(conv_output, bias)
    norm_output = _instance_norm(conv_output)
    if use_relu:
        output = tf.nn.relu(norm_output)
        return output
    else:
        return norm_output

# def _conv_layer_transposed(inputs, num_filters, filter_shape, stride):
    # kernel = _init_kernel(inputs, num_filters, filter_shape)
    # conv_output = tf.nn.conv2d_transpose(inputs, filter=kernel, strides=[1, stride, stride, 1], padding='SAME')
    # bias = tf.get_variable(name='bias', shape=[num_filters], initializer = tf.zeros_initializer())
    # conv_output = tf.nn.bias_add(conv_output, bias)
    # norm_output = _instance_n

def _residual_block(inputs):
    block_conv = _conv_layer(inputs, 128, 3, 1)
    return inputs + _conv_layer(block_conv, 128, 3, 1, False)

def _instance_norm(inputs):
    means = tf.reduce_mean(inputs, axis=[1, 2])
    stddevs = tf.reduce_std(inputs, axis=[1, 2])
    instance_norm = (inputs - means)/stddevs
    return instance_norm

def _instance_norm(inputs):
    means, stddevs = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
    return (inputs - means)/stddevs

if __name__ == "__main__":
    random_vals = np.random.random(size = [4, 252, 252, 3])
    sess = tf.Session()
    x = tf.constant(random_vals, dtype=tf.float32)
    output = transform_net(x) 
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer())
    o = sess.run(output)
    
