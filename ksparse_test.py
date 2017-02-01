# Test ksparse-autoencoder on dummy data

import tensorflow as tf
import numpy as np
import ksparse_base


if __name__ == '__main__':

    # Defining fixed params
    VEC_LENGTH = 4
    NUM_ATOMS = 2

    TRAIN_BATCH_SIZE = 4

    inp_data = np.array([[1, 1, 0, 0], [0, 1, 0, 1], [0.9, 1.1, 0, 0], [
                        0, 0.8, 0, 0.9]], dtype=float)

    x = tf.placeholder(tf.float32, [None, VEC_LENGTH], name='x')
    max_nonzero = tf.placeholder(tf.int32, [], name='max_nonzero')

    # W = ksparse_base.weight_variable([VEC_LENGTH, NUM_ATOMS])
    W = tf.Variable([[1, 0], [1, 1], [0, 0], [0, 1]], dtype=tf.float32)
    b = ksparse_base.bias_variable([NUM_ATOMS, 1])
    b_ = ksparse_base.bias_variable([VEC_LENGTH, 1])

    y, W = ksparse_base.ksparse_autoenc_withbias(x, W, b, b_, max_nonzero)

    # Defining L2 loss
    loss = tf.nn.l2_loss(x - y, name='reconstruction_error')
    tf.summary.scalar('L2 loss', loss)
    # Train using optimizer
    train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(loss)

    merged_summary = tf.summary.merge_all()

    # Learning

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    # Adding summary writers for Tensorflow
    train_writer = tf.summary.FileWriter('train', sess.graph)
    test_writer = tf.summary.FileWriter('test', sess.graph)

    NUM_NONZERO = 1

    summary, _ = sess.run([merged_summary, loss], feed_dict={
                          x: inp_data, max_nonzero: NUM_NONZERO})

    train_writer.add_summary(summary, 0)

    """
    Run
    """

    for summ_step in range(1, 2000):
        summary, _ = sess.run([merged_summary, loss],
                              feed_dict={x: inp_data,
                                         max_nonzero: NUM_NONZERO})

        train_writer.add_summary(summary, summ_step)

    print(sess.run(W))
    print(b.eval(session=sess))
    print(b_.eval(session=sess))
    print(sess.run(y, feed_dict={x: inp_data, max_nonzero: NUM_NONZERO}))

    sess.close()
