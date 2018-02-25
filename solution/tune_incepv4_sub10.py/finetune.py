from __future__ import division
from __future__ import print_function

from os.path import join
import gc
gc.disable()
import tensorflow as tf

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

    with tf.name_scope('FineTuning'):
        # flattened = tf.concat([aux, avg], axis=-1)
        flattened = avg
        logits = slim.fully_connected(flattened, 1024, activation_fn=None,
                                        scope='FineTuningLayer')
        logits = tf.nn.relu(logits)
        logits = slim.fully_connected(logits, num_classes, activation_fn=None,
                                        scope='FineTuningLogits')
    predictions = tf.nn.softmax(logits, name='output')

    return logits, predictions

def _finetune():
    with tf.Session(config=config) as sess:
        # view checkpoint
        # print_tensors_in_checkpoint_file(ckpt_pth, None, False)
        
        batch_size = 64
        num_classes = 200
        epoch = datatool.num_train//batch_size

        print('Rebuilding model...')
        incep_vars, end_points = rebuild_incepv4([batch_size, size, size, 3], training=False, dropout_keep_prob=.9)
        
        print('Attaching new final layer...')
        logits, predictions = add_fine_tuning_parts(end_points, num_classes)

        print('Setting up loss function...')
        cost = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.placeholder(dtype=tf.float32, shape=[batch_size, 200], name='labels'),
            logits=logits
        )

        last_layers = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FineTuning')
        last_layers += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'InceptionV4')
        removeable = []
        for i, layer in enumerate(last_layers):
            ll = str(layer)
            if ('FineTuning' not in ll) and ('Mixed_7' not in ll): # only the last several layers
                removeable.append(layer)
            # elif 'BatchNorm' in ll:
            #     removeable.append(layer)
        for i in removeable:
            last_layers.remove(i)

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
        new_layers = initialize("FineTuning")

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


        # set up valitaion error checking function
        def val_loss(batch_start):
            val = datatool.get_val(batch_size)
            data, summary = sess.run([loss_data, log_val], feed_dict={'input:0':val[0], 'labels:0': val[1]})
            writer.add_summary(summary, batch_start)
            return data

        global val_loss_saved
        val_loss_saved = 1000
        # set up training function
        def train(batch_start, num_batches, learning_rate, new_layers_only=True):
            global val_loss_saved
            last = 2
            with tf.variable_scope('Optimizer'  +str(batch_start)):
                optimizer = tf.train.AdamOptimizer(learning_rate)
                if new_layers_only:
                    train_op = optimizer.minimize(cost, var_list=new_layers)
                else:
                    train_op = optimizer.minimize(cost, var_list=last_layers)
            opt = initialize("Optimizer" + str(batch_start))

            # do training loop
            for i in range(num_batches):
                batch = datatool.get_train_batch(batch_size)
                _lossvalue, _, summary = sess.run([loss_data, train_op, log_train], feed_dict={'input:0':batch[0], 'labels:0': batch[1]})
                if (i%10)==0:
                    print(i, '/', epoch, _lossvalue)
                    writer.add_summary(summary, batch_start + i)
                if (i % 1000) == 0:
                    val_loss_current = val_loss(i+batch_start)
                    print(i, 'val_loss:',val_loss_current, 'saved:', val_loss_saved)
                    if (val_loss_current < val_loss_saved): # saves best
                        val_loss_saved = val_loss_current
                        print('******saving best')
                        fine_tuned_saver.save(sess, join(save_path, 'inception_v4_299_tuned_ncs_BEST{}.ckpt'.format(int(val_loss_current*10000))))
                

        # Start training
        print('training stage 0...')
        pos = 0
        num = epoch
        # first start by training one epoch
        # and updating weights for only the last fully connected layer.
        train(pos, num, 0.001, True)
        pos += num


        schedule = [0.0001, 0.000093, 0.000084, 0.000073]
        for i, lr in enumerate(schedule):
            print('training stage {}...'.format(i+1))
            train(pos, num, lr, False)
            pos += num
        
        return join(save_path, 'inception_v4_299_tuned_ncs_BEST{}.ckpt'.format(int(val_loss_saved*10000)))

def _save_meta(path):
    finetuned_path = path
    output_meta_path = join('incepv4', 'finetuned', 'meta', 'network')

    num_classes = 200
    size = 299
    batch_size = 1

    # Load origional inceptionv4 model
    incep_vars, end_points = rebuild_incepv4([batch_size, size, size, 3], training=False, dropout_keep_prob=1) # do not add dropout layers


    # add finetuning computations
    logits, predictions = add_fine_tuning_parts(end_points, num_classes)

    with tf.Session() as sess:

        saver = tf.train.Saver()

        saver.restore(sess, finetuned_path)

        # Test execution of the model
        preds = sess.run(predictions, feed_dict={'input:0': np.random.random([batch_size, size, size, 3])})
        print(preds)

        # save model and weights in meta folder
        saver.save(sess, output_meta_path)
        saver.export_meta_graph(
        filename=output_meta_path + '.meta',
        clear_devices=True,
        clear_extraneous_savers=True)

if __name__ == "__main__":
    best_path = _finetune()
    _save_meta(best_path)


    
    


