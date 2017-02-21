import tensorflow as tf

checkpoint = "./checkpoints/rnn_train_1487158682-1500000"


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./checkpoints/rnn_train_1487158682-1500000.meta')
    new_saver.restore(sess, checkpoint)