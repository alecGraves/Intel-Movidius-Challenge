
from os.path import join

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
slim = tf.contrib.slim

# data tool
import datatool

# model specific info
from incepv4.inception_v4 import inception_v4
from incepv4.inception_utils import inception_arg_scope as scope
size = inception_v4.default_image_size
ckpt_pth = join('incepv4', 'weights', 'inception_v4.ckpt')

def rebuild_incepv4(input_shape=[1, size, size, 3], training=False):
    # build the model, get loadable vars
    with slim.arg_scope(scope()):
        input_layer = tf.placeholder(shape=input_shape, dtype=tf.float32, name='input')
        logits, end_points = inception_v4(input_layer, is_training=training)
        vars_to_restore = slim.get_variables_to_restore()

    # save the op used to restore weights from checkpoint
    restore_op = tf.train.Saver(vars_to_restore).restore
    return input_layer, restore_op, end_points

def add_fine_tuning_parts(end_points, num_classes=200):
    # grab the network output
    flatten_avg_pool = end_points['PreLogitsFlatten']
    flatten_aux = end_points['AuxFlatten']

    # concatenate, add fc layer
    flattened = tf.concat(axis=-1, values=[flatten_aux, flatten_avg_pool])
    with tf.variable_scope('FineTuning'):
        logits = slim.fully_connected(flattened, num_classes, activation_fn=None,
                                        scope='FineTuning')
        predictions = tf.nn.softmax(logits, name='PredictionsFine')

    return logits, predictions

if __name__ == "__main__":
    with tf.Session() as sess:
        # view checkpoint
        # print_tensors_in_checkpoint_file(ckpt_pth, None, False)
        
        batch_size = 15
        num_epochs = 100
        num_classes = 200

        # build model
        input_layer, restore_op, end_points = rebuild_incepv4([batch_size, size, size, 3], training=True)
        
        # load the weights
        restore_op(sess, ckpt_pth)

        # attach new output layer
        logits, predictions = add_fine_tuning_parts(end_points, num_classes)

        #attach loss function
        cost = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.placeholder(dtype=tf.float32, shape=[batch_size, 200], name='labels'),
            logits=logits
        )

        ### Fine-Tune ###
        # grab variables to fine-tune
        trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FineTuning")
        for var in trainable:
            sess.run(var.initializer)
    
        # define training ops
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train_op = optimizer.minimize(cost, var_list=trainable)

        for i in range(1000):
            batch = datatool.get_train_batch(batch_size)
            _, loss = sess.run([train_op, cost], feed_dict={'input:0':batch[0], 'labels:0': batch[1]})
            print(sum(loss)/len(loss))




    
    


