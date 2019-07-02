import os
import tensorflow as tf

from transform_net import transform_net
from utils import load_image, save_image

def transfer_style(image_path, checkpoint_path, output_path):

    image = load_image(image_path, expand_dims=True)

    input_image = tf.placeholder(tf.float32, shape=[1, 256, 256, 3])
    style_output = transform_net(input_image)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_path):
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                # from tensorflow.contrib.framework.python.framework import checkpoint_utils
                # var_list = checkpoint_utils.list_variables(ckpt.model_checkpoint_path)
                # for v in var_list:
                #     print(v)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No Checkpoint Found")
        else:
            saver.restore(sess, checkpoint_path)
        stylized_image = sess.run(style_output, feed_dict={input_image:image})

    save_image(output_path, stylized_image[0])

if __name__ == '__main__':
    img_path = '../images/content/puppy.jpg'
    out_path = '../images/stylized/style_puppy_{}.jpg'.format(115)
    ckpt_path = 'checkpoints'
    transfer_style(img_path, ckpt_path, out_path)






