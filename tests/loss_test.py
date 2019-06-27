import numpy as np
import sys
import tensorflow as tf
import unittest

sys.path.append('../models')
from train import layer_content_loss, layer_style_loss

def np_content_loss(pred, target):
    size = pred.size
    return np.sum((pred - target)**2) / size

def np_style_loss(pred, target):
    batch_size = pred.shape[0]
    channels = pred.shape[-1]
    input_size = pred.size
    reshape_pred = np.reshape(pred, [batch_size, channels, -1])
    reshape_pred_t = np.reshape(pred, [batch_size, -1, channels])
    reshape_target = np.reshape(target, [batch_size, channels, -1])
    reshape_target_t = np.reshape(target, [batch_size, -1, channels])
    gram_pred = np.matmul(reshape_pred, reshape_pred_t)
    gram_target = np.matmul(reshape_target, reshape_target_t)
    return np.sum((gram_pred-gram_target)**2) / (input_size ** 2) / batch_size

def total_variation(image):
    w_shift, h_shift = image[:,1:,:,:], image[:,:,1:,:]
    w_shape, h_shape = w_shift.size, h_shift.size
    w_shift_diff = w_shift - image[:,:-1,:,:]
    h_shift_diff = h_shift - image[:,:,:-1,:]
    w_total_var = np.sqrt(np.sum(w_shift_diff ** 2))
    h_total_var = np.sqrt(np.sum(h_shift_diff ** 2))
    return 2 * (w_total_var / w_shape + h_total_var / h_shape)


class Test(unittest.TestCase):

    def test_content_loss(self):
        x = np.random.random([4, 10, 10, 3])
        y = np.random.random([4, 10, 10, 3])
        np_result = np_content_loss(x, y)

        with tf.Session() as sess:
            tensor_x = tf.constant(x, dtype=tf.float32)
            tensor_y = tf.constant(y, dtype=tf.float32)
            tf_result = sess.run(layer_content_loss(tensor_x, tensor_y))

        tf.reset_default_graph()
        success = np.allclose(np_result, tf_result)
        self.assertEqual(success, True)

    def test_style_loss(self):
        x = np.random.random([4, 256, 256, 3])
        y = np.random.random([4, 256, 256, 3])
        np_result = np_style_loss(x, y)

        with tf.Session() as sess:
            tensor_x = tf.constant(x, dtype=tf.float32)
            tensor_y = tf.constant(y, dtype=tf.float32)
            tf_result = sess.run(layer_style_loss(tensor_x, tensor_y))

        tf.reset_default_graph()
        success = np.allclose(np_result, tf_result)
        self.assertEqual(success, True)

if __name__ == '__main__':
    unittest.main()
