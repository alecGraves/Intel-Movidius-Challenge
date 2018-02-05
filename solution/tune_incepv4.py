from __future__ import division
from __future__ import print_function

from os.path import join

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
slim = tf.contrib.slim
tf.set_random_seed(751)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# data tool
import datatool

# model specific info
from incepv4.inception_v4 import inception_v4
from incepv4.inception_utils import inception_arg_scope as scope
size = 299
ckpt_pth = join('incepv4', 'weights', 'inception_v4.ckpt')

debug = True

def rebuild_incepv4(input_shape=[1, size, size, 3], training=False, dropout_keep_prob=1):
    # build the model, get loadable vars
    with slim.arg_scope(scope()):
        input_layer = tf.placeholder(shape=input_shape, dtype=tf.float32, name='input')
        logits, end_points = inception_v4(input_layer, is_training=training, create_aux_logits=True, dropout_keep_prob=dropout_keep_prob)
        vars_to_restore = slim.get_variables_to_restore()

    # save the op used to restore weights from checkpoint
    return vars_to_restore, end_points

def add_fine_tuning_parts(end_points, num_classes=200):
    # grab the network output
    aux = end_points['AuxFlatten']
    avg = end_points['PreLogitsFlatten']

    with tf.variable_scope('FineTuning'):
        flattened = tf.concat([aux, avg], axis=-1)
        logits = slim.fully_connected(flattened, num_classes, activation_fn=None,
                                        scope='FineTuning')
        predictions = tf.nn.softmax(logits, name='PredictionsFine')
    output = tf.identity(logits, name='logits')

    return logits, predictions

if __name__ == "__main__":
    with tf.Session(config=config) as sess:
        # view checkpoint
        # print_tensors_in_checkpoint_file(ckpt_pth, None, False)
        
        batch_size = 64
        num_classes = 200

        if debug:
            print('Rebuilding model...')
        incep_vars, end_points = rebuild_incepv4([batch_size, size, size, 3], training=False)
        
        if debug:
            print('Attaching new final layer...')
        logits, predictions = add_fine_tuning_parts(end_points, num_classes)

        if debug:
            print('Setting up loss function...')
        cost = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.placeholder(dtype=tf.float32, shape=[batch_size, 200], name='labels'),
            logits=logits
        )

        restore_op = tf.train.Saver(incep_vars).restore
        restore_op(sess, ckpt_pth)
        print('done loading weights')
        #################
        ### Fine-Tune ###
        #################
        def initialize(var_scope):
            scope_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_scope)
            scope_vars += tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, var_scope)
            # print(scope_vars)
            for var in scope_vars:
                sess.run(var.initializer)
            return scope_vars

        # 'Initializing new weights for training...'
        trainable = initialize("FineTuning")

        # set up logging functionality
        with tf.variable_scope('Logging'):
            logs_path = join('incepv4', 'finetuned', 'logs')
            loss_data = tf.reduce_mean(cost)
            log_val = tf.summary.scalar("val_cross_entropy", loss_data)
            log_train = tf.summary.scalar("train_cross_entropy", loss_data)
            writer =  tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        initialize('Logging')
        # test save
        save_path =  join('incepv4', 'finetuned', 'weights')
        fine_tuned_saver = tf.train.Saver()
        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_save_test.ckpt'))

        # set up valitaion error checking function
        def val_loss(batch_start):
            val = datatool.get_val(batch_size=batch_size)
            data, summary = sess.run([loss_data, log_val], feed_dict={'input:0':val[0], 'labels:0': val[1]})
            writer.add_summary(summary, batch_start)
            return data
        
        val_loss_record = []
        # set up training function
        def train(batch_start, num_batches, learning_rate):
            with tf.variable_scope('Optimizer'  +str(batch_start)):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(cost, var_list=trainable)
            opt = initialize("Optimizer" + str(batch_start))

            # do training loop
            for i in range(num_batches):
                batch = datatool.get_train_batch(batch_size)
                _, summary = sess.run([train_op, log_train], feed_dict={'input:0':batch[0], 'labels:0': batch[1]})
                writer.add_summary(summary, batch_start + i)
                if i == 0 or(i % 10) == 0:
                    val_loss_record.append(val_loss(i+batch_start))
                    print(i, 'val_loss:', val_loss_record[-1])
                    if i > 300 and val_loss_record[-1] < min(val_loss_record[:-1]):
                        print('saving best')
                        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_299_tuned_ncs_BEST.ckpt'), )
                

        # Start training
        epoch = datatool.num_train//batch_size
        print('training stage 1...')
        train(0, epoch//2, learning_rate=0.001)

        print('training stage 2...')
        train(epoch//2, epoch//2, learning_rate=0.0001)

        print('training stage 3...')
        train(epoch, epoch//2, learning_rate=0.000093)

        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_299_tuned_ncs.ckpt'))




    
    


