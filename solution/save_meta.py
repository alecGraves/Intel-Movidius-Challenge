from os.path import join
import tensorflow as tf
import numpy as np
from tune_incepv4 import rebuild_incepv4, add_fine_tuning_parts, scope, slim, inception_v4

finetuned_path =  join('incepv4', 'finetuned', 'weights', 'inception_v4_299_tuned_ncs.ckpt')
output_meta_path = join('incepv4', 'finetuned', 'meta', 'model')

# Load origional inceptionv4 model
with slim.arg_scope(scope()):
    input_layer = tf.placeholder(shape=[1, 299, 299, 3], dtype=tf.float32, name='input')
    logits, end_points = inception_v4(input_layer, is_training=False)
    vars_to_restore = slim.get_variables_to_restore()
flatten_avg_pool = end_points['PreLogitsFlatten']
flatten_aux = end_points['AuxFlatten']

# add finetuning computations
flattened = tf.concat(axis=-1, values=[flatten_aux, flatten_avg_pool])
with tf.variable_scope('FineTuning'):
    logits = slim.fully_connected(flattened, 200, activation_fn=None,
                                    scope='FineTuning')
    predictions = tf.nn.softmax(logits, name='PredictionsFine')
 
with tf.Session() as sess:
    # restore vars from checkpoint
    fine_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'FineTuning')
    vars = vars_to_restore + fine_vars
    saver = tf.train.Saver(vars)

    saver.restore(sess, finetuned_path)

    # Test execution of the model
    preds = sess.run(predictions, feed_dict={'input:0': np.random.random([1, 299, 299, 3])})
    print(preds)    

    # save model in meta folder
    saver.save(sess, output_meta_path)

print('run "mvNCCompile {0} -in=input -on=FineTuning/PredictionsFine -s12" to compile'.format(output_meta_path+'.meta'))