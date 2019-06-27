import numpy as np
import sys
import tensorflow as tf
import unittest

sys.path.append('../models')
from transform_net import _conv_layer

class Test(unittest.TestCase):

    def test_conv_1_shape(self):
        x = np.random.random([4, 256, 256, 3])
        tensor_x = tf.constant(x, tf.float32)
        conv_1 = _conv_layer(tensor_x, 32, 9, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_result = sess.run(conv_1)

        tf.reset_default_graph()
        self.assertEqual(tf_result.shape, (4, 256, 256, 32))

    def test_conv_2_shape(self):
        x = np.random.random([4, 256, 256, 3])
        tensor_x = tf.constant(x, tf.float32)
        conv_1 = _conv_layer(tensor_x, 32, 9, 1)
        conv_2 = _conv_layer(conv_1, 64, 3, 2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf_result = sess.run(conv_2)

        tf.reset_default_graph()
        self.assertEqual(tf_result.shape, (4, 128, 128, 64))

if __name__ == '__main__':
    unittest.main()
