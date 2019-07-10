import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import time

from eval import transfer_style
from transform_net import transform_net
from loss_net import vgg, normalize, denormalize
from utils import load_image, image_generator, save_image

train_data_path = '../data/train_data'
content_layer = 'relu4_2'
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

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
    # input_shape = tf.shape(pred_gram, out_type=tf.float32)
    # batch_size = input_shape[0]
    input_size = tf.size(pred_gram, out_type=tf.float32)
    return tf.reduce_sum((pred_gram-target_gram)**2) / input_size

def gram(batch_input):
    """
    Reshapes the batch input to the proper shape for efficient calculation
    of the Gram matrix and then returns the Gram matrix.
    """
    input_shape = tf.shape(batch_input, out_type=tf.float32)
    batch_size, h, w, filters = (input_shape[i] for i in range(4))
    reshape_batch = tf.reshape(batch_input, [batch_size, filters, h*w])
    reshape_batch_t = tf.reshape(batch_input, [batch_size, h*w, filters])
    return tf.matmul(reshape_batch, reshape_batch_t) / (h * w * filters)


def tv_loss(image, batch_size):
    """
    Calculates the total variation loss for the output of the transform net
    """
    tv_y_size = tf.size(image[:,1:,:,:], out_type=tf.float32)
    tv_x_size = tf.size(image[:,:,1:,:], out_type=tf.float32)
    y_tv = tf.reduce_sum((image[:,1:,:,:] - image[:,:-1,:,:])**2)
    x_tv = tf.reduce_sum((image[:,1:,:,:] - image[:,:-1,:,:])**2)
    return (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

def optimize(style_name, style_path, epochs, batch_size, learning_rate, style_w,
             content_w, tv_w, save_step, checkpoint_path,
             test_image_name, test_image_path, eval_step, debug=False):
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
    with tf.name_scope("style_comp"):
        style_image_norm = normalize(style_image)
        style_act_dict = vgg(style_image_norm)
        style_gram_dict = {}
        with tf.Session() as sess:
            style_content_layer = sess.run(style_act_dict[content_layer])
            for key, act in style_act_dict.items():
                style_act_dict[key] = sess.run(style_act_dict[key])
            for layer in style_layers:
                style_gram_dict[layer] = sess.run(gram(style_act_dict[layer]))
    # return style_act_dict, style_gram_dict

    if debug:
        for layer, act in style_act_dict.items():
            # Save the first 16 activations
            fig, axes = plt.subplots(4, 4)
            for i in range(16):
                j, k = i % 4, i // 4
                axes[j, k].imshow(act[0][:,:,i])
            fig.suptitle("Activations for Layer: {}".format(layer))
            plt.show()
            # save_image(act, '../images/layers/{}_{}.jpg'.format(style_name, layer))
        for layer, gram_ in style_gram_dict.items():
            # save_image(gram_, '../images/layers/{}_{}_gram.jpg'.format(style_name, layer))
            fig, axes = plt.subplots(4, 4)
            for i in range(16):
                j, k = i % 4, i // 4
                axes[j, k].imshow(gram_[0][:,:])
            fig.suptitle("Visualize Gram for Layer: {}".format(layer))
            plt.show()

    # Compute the content image activations
    input_image = tf.placeholder(tf.float32, shape=[batch_size, 256, 256, 3], name='image_input')
    input_image_norm = normalize(input_image)
    input_content_layer = vgg(input_image_norm)[content_layer]

    output_image = transform_net(input_image/255.0)
    norm_output_image = normalize(output_image)
    output_act_dict = vgg(norm_output_image)
    output_gram_dict = {}
    for key, act in output_act_dict.items():
        output_gram_dict[key] = gram(act)

    # calculate the losses
    style_losses = []
    for l in style_layers:
        style_losses.append(layer_style_loss(output_gram_dict[l], style_gram_dict[l]))
    total_var_loss = tv_loss(output_image, batch_size)
    style_loss = tf.add_n(style_losses) / batch_size
    content_loss = layer_content_loss(input_content_layer, output_act_dict[content_layer])
    loss = style_w * style_loss + content_w * content_loss + tv_w * total_var_loss
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("summaries/")
        writer.add_graph(sess.graph)

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        # if os.path.isdir(checkpoint_path):
        #     ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #     else:
        #         raise Exception("No Checkpoint Found")
        # else:
        #     saver.restore(sess, checkpoint_path)
        for i in range(epochs):
            start_time = time.time()
            for batch in image_generator('../data/train2014', batch_size):
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
                    saver.save(sess, "checkpoints/model", step)
                    start_time = time.time()

                # As a test during the training time, record evals
                # of the various models to make sure training is
                # happening
                if step % eval_step == 0:
                    output_path = '../images/stylized/{}_{}.jpg'.format(test_image_name, step)
                    test_image = load_image(test_image_path)
                    test_batch = np.array([test_image for i in range(batch_size)])
                    test_image_out = sess.run(output_image,
                                              feed_dict={input_image:test_batch})
                    save_image(output_path, test_image_out[0, :, :, :])


if __name__ == '__main__':
    content_w = 7.5e0
    style_w = 1e2
    tv_w = 2e2

    style_image_path = '../images/style/big_wave.jpg'
    # style_image = load_image(style_image_path, expand_dims=True)
    # style_input = tf.constant(style_image, tf.float32)
    optimize('wave', style_image_path, 1, 4, 1e-3,
             style_w, content_w, tv_w, 3000, 'checkpoints',
             'hearst', '../images/content/hearst_mining.jpg', 1000, debug=False)

    # dict1, dict2 = optimize('princess', style_image_path, 1, 4, 1e-3,
             # style_w, content_w, tv_w, 300, 'checkpoints',
             # 'hearst', '../images/content/hearst_mining.jpg', 100, debug=True)




