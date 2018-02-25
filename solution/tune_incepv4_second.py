from __future__ import division
from __future__ import print_function

from os.path import join

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
slim = tf.contrib.slim
tf.set_random_seed(94613)

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
        logits, end_points = inception_v4(input_layer, is_training=training, create_aux_logits=False, dropout_keep_prob=dropout_keep_prob)
        vars_to_restore = slim.get_variables_to_restore()

    # save the op used to restore weights from checkpoint
    return vars_to_restore, end_points

def add_fine_tuning_parts(end_points, num_classes=200):
    # grab the network output
    avg = end_points['PreLogitsFlatten']
    # aux = end_points['AuxFlatten']

    with tf.variable_scope('FineTuning'):
        # flattened = tf.concat([aux, avg], axis=-1)
        flattened = avg
        logits = slim.fully_connected(flattened, num_classes, activation_fn=None,
                                        scope='FineTuning')
        predictions = tf.nn.softmax(logits, name='PredictionsFine')
    output = tf.identity(logits, name='logits')

    return logits, predictions

if __name__ == "__main__":
    with tf.Session(config=config) as sess:
        # view checkpoint
        # print_tensors_in_checkpoint_file(ckpt_pth, None, False)
        
        batch_size = 8
        num_classes = 200

        if debug:
            print('Rebuilding model...')
        incep_vars, end_points = rebuild_incepv4([batch_size, size, size, 3], training=False, dropout_keep_prob=.9)
        
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
        fine_tuned_saver = tf.train.Saver(max_to_keep=None)
        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_save_test.ckpt'))

        fine_tuned_saver.restore(sess, join(save_path, 'run7', 'inception_v4_299_tuned_ncs_BEST661.ckpt'))

        # set up valitaion error checking function
        def val_loss(batch_start):
            val = datatool.get_val()
            data = []
            for i in range(len(val[0])//batch_size):
                idx_s = i*batch_size
                idx_e = i*batch_size+batch_size
                d, summary = sess.run([loss_data, log_val], feed_dict={'input:0':val[0][idx_s:idx_e], 'labels:0': val[1][idx_s:idx_e]})
                data.append(d)
            # writer.add_summary(summary, batch_start)
            return sum(data)/len(data)

        
        # set up training function
        def train(batch_start, num_batches, learning_rate, trainable_only=True):
            val_loss_saved = 1000
            last = 2
            with tf.variable_scope('Optimizer'  +str(batch_start)):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                if trainable_only:
                    train_op = slim.learning.create_train_op(cost, optimizer)
                    # train_op = optimizer.minimize(cost, var_list=trainable)
                    # slim.create_train_op
                else:
                    train_op = slim.learning.create_train_op(cost, optimizer)
            opt = initialize("Optimizer" + str(batch_start))

            # do training loop
            for i in range(num_batches):
                batch = datatool.get_train_batch(batch_size)
                _, summary = sess.run([train_op, log_train], feed_dict={'input:0':batch[0], 'labels:0': batch[1]})
                writer.add_summary(summary, batch_start + i)
                if (i % 20) == 0:
                    val_loss_current = val_loss(i+batch_start)
                    print(i, 'val_loss:',val_loss_current, 'saved:', val_loss_saved)
                    last += 1
                    if (val_loss_current < val_loss_saved) and (last > 2): # saves seperated by at least 2 validations
                        last = 0
                        val_loss_saved = val_loss_current
                        print('saving best')
                        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_299_tuned_ncs_BEST{}.ckpt'.format(int(val_loss_current*1000))))


        # Start training
        epoch = datatool.num_train//batch_size
        print('training stage 0...')
        pos = 0

        # num = epoch//2
        # train(pos, num, learning_rate=0.001)
        # pos += num

        num = epoch

        # schedule = [0.0001, 0.000093, 0.000084, 0.000073]
        schedule = [0.0001, 0.000093, 0.000084, 0.000073]
        for i, lr in enumerate(schedule):
            print('training stage {}...'.format(i+1))
            train(pos, num, lr, False)
            pos += num

        # for i, lr in enumerate(schedule):
        #     print('training stage {}...'.format(i+1))
        #     train(pos, num, learning_rate=0.0001, False)
        #     pos += num

        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_299_tuned_ncs.ckpt'))




    
    


