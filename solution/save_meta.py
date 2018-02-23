from os.path import join
import tensorflow as tf
import numpy as np
from tune_incepv4 import rebuild_incepv4, add_fine_tuning_parts, scope, slim, inception_v4

finetuned_path =  join('incepv4', 'finetuned', 'weights', 'run7', 'inception_v4_299_tuned_ncs_BEST0.6397403478622437.ckpt')
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
