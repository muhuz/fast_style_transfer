import numpy as np
import os
import tensorflow as tf
import time

from transform_net import transform_net
from loss_net import vgg, normalize, denormalize
from utils import load_image, image_generator

train_data_path = '../data/train_data'
content_layer = 'relu2_2'
style_layers = ['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4']

def layer_content_loss(pred, target):
    """
    Calculates the feature reconstruction loss for feature maps.
    """
    input_size = tf.size(pred, out_type=tf.float32)
    return 1 / input_size * tf.reduce_sum((pred - target)**2)

def layer_style_loss(pred_gram, target_gram):
    """
    Calculates the style reconstruction loss for the inputed gram matrices
    using the Gram Matrix.
    """
    input_shape = tf.shape(pred_gram, out_type=tf.float32)
    batch_size = input_shape[0]
    input_size = tf.size(pred_gram, out_type=tf.float32)
    # reshape_pred = tf.reshape(pred, [batch_size, channels, -1])
    # reshape_pred_t = tf.reshape(pred, [batch_size, -1, channels])
    # reshape_target = tf.reshape(target, [batch_size, channels, -1])
    # reshape_target_t = tf.reshape(target, [batch_size, -1, channels])
    # gram_pred = tf.matmul(reshape_pred, reshape_pred_t)
    # gram_target = tf.matmul(reshape_target, reshape_target_t)
    return tf.reduce_sum((pred_gram-target_gram)**2) / (input_size ** 2) / batch_size

def gram(batch_input):
    """
    Reshapes the batch input to the proper shape for efficient calculation
    of the Gram matrix and then returns the Gram matrix.
    """
    input_shape = tf.shape(batch_input, out_type=tf.float32)
    batch_size, channels = input_shape[0], input_shape[-1]
    reshape_batch = tf.reshape(batch_input, [batch_size, channels, -1])
    reshape_batch_t = tf.reshape(batch_input, [batch_size, -1, channels])
    return tf.matmul(reshape_batch, reshape_batch_t)


def tv_loss(image, batch_size):
    """
    Calculates the total variation loss for the output of the transform net
    """
    tv_y_size = tf.size(image[:,1:,:,:], out_type=tf.float32)
    tv_x_size = tf.size(image[:,:,1:,:], out_type=tf.float32)
    y_tv = tf.reduce_sum((image[:,1:,:,:] - image[:,:-1,:,:])**2)
    x_tv = tf.reduce_sum((image[:,1:,:,:] - image[:,:-1,:,:])**2)
    return (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

def optimize(style_path, epochs, batch_size, learning_rate, style_w,
    content_w, tv_w, save_step):
    """
    input a list of file names to batch into the model
    """
    style_image = load_image(style_path, expand_dims=True)
    # style_input = tf.constant(style_image, tf.float32)

    # Compute the outputs for the style image that will be
    # used for the calculation of the loss function. This
    # includes the gram matricies of the activations used
    # for style loss and the activations of the layer used for
    # content loss.
    style_image_norm = normalize(style_image)
    style_act_dict = vgg(style_image_norm)
    style_gram_dict = {}
    with tf.Session() as sess:
        style_content_layer = sess.run(style_act_dict[content_layer])
        for key, act in style_act_dict.items():
            style_gram_dict[key] = sess.run(style_act_dict[key])
        for layer in style_layers:
            style_gram_dict[layer] = sess.run(gram(style_act_dict[layer]))

    # Compute the content image activations
    input_image = tf.placeholder(tf.float32, shape=[batch_size, 256, 256, 3])
    input_image_norm = normalize(input_image)
    output_image = transform_net(input_image_norm)
    output_act_dict = vgg(output_image)
    output_gram_dict = {}
    for key, act in output_act_dict.items():
        output_gram_dict[key] = gram(act)

    style_losses = []
    for l in style_layers:
        style_losses.append(layer_style_loss(output_gram_dict[l], style_gram_dict[l]))

    # calculate the losses
    total_var_loss = tv_loss(output_image, batch_size)
    style_loss = tf.add_n(style_losses) / batch_size
    content_loss = layer_content_loss(output_act_dict['relu2_2'], style_act_dict['relu2_2'])
    loss = style_w * style_loss + content_w * content_loss + tv_w * total_var_loss

    saver = tf.train.Saver()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epochs):
            start_time = time.time()
            for batch in image_generator('../data/train2014', batch_size):
                print(batch.shape)
                print(batch.dtype)
                feed_dict = {input_image:batch}
                optimizer.run(feed_dict=feed_dict)
                step = global_step.eval()
                if step % save_step == 0:
                    loss_list = [style_loss, content_loss, total_var_loss, loss]
                    losses = sess.run(loss_list, feed_dict=feed_dict)
                    seconds = time.time() - start_time
                    print('Step {}\n   Loss: {:5.1f}'.format(step, losses[3]))
                    print('   Style Loss: {:5.1f}'.format(losses[0]))
                    print('   Content Loss: {:5.1f}'.format(losses[1]))
                    print('   TV Loss: {:5.1f}'.format(losses[2]))
                    print('   Took: {} seconds'.format(seconds))
                    saver.save(sess, "checkpoints/model.ckpt")
                    start_time = time.time()

if __name__ == '__main__':
    content_w = 7.5e0
    style_w = 1e2
    tv_w = 2e2

    style_image_path = '../images/style/abstract_rainbow.jpg'
    # style_image = load_image(style_image_path, expand_dims=True)
    # style_input = tf.constant(style_image, tf.float32)
    optimize(style_image_path, 1, 4, 1e-3, style_w, content_w, tv_w, 10)





