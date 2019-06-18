import numpy as n
import os
import tensorflow as tf

import transform_net
from loss_model import Vgg

EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.01

train_data_path = ''
style_image_path = ''
content_layer = 'relu2_2' 
style_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

def feature_loss(pred, target):
    """
    Calculates the feature reconstruction loss for feature maps.
    """
    input_size = tf.size(pred)
    return 1 / input_size * tf.norm(pred - target)

def style_loss(pred, target):
    """
    Calculates the style reconstruction loss for feature maps
    using the Gram Matrix.
    """
    in_channels = tf.shape(pred)
    input_size = tf.size(pred)
    reshape_pred = tf.reshape(pred, [in_channels, -1])
    reshape_target = tf.reshape(target, [in_channels, -1])
    return 1/(input_size ** 2) * tf.norm(reshape_pred-reshape_target)

def loss()

dataset = tf.data.Dataset.from_tensor_slices()
batch_dataset = dataset.batch(4)
iterator = batch_dataset.make_one_shot_iterator()

with tf.Session() as sess:
    for i in range(EPOCHS):
        batch = sess.run(iterator.get_next())
        sess.run([getattr(vgg, layer) for layer in style_layers])
        sess.run(style_loss(layer) for layer in style_layers)

