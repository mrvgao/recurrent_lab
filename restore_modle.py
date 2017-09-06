import tensorflow as tf

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/mean_model_length_10.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))
    graph = tf.get_default_graph()

    train_X = graph.get_tensor_by_name('X:0')
    y_hat = graph.get_tensor_by_name('y_hat:0')

    test_x = [
        [[1], [20], [3], [4], [5], [6], [7], [8], [9], [100]]
    ]

    _y_hat = sess.run(y_hat, feed_dict={train_X: test_x})
    print(_y_hat)

