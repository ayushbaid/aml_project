# Train ksparse-autoencoder on MNISt

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from datetime import datetime
from scipy.io import savemat
from scipy.misc import imsave
import ksparse_base
import os


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/")
    now = datetime.now()

    # Defining fixed params
    IM_EDGE = 28
    IM_CHANNELS = 1
    VEC_LENGTH = (IM_EDGE**2) * IM_CHANNELS

    NUM_ATOMS = 300

    save_path_prefix = 'mnist/'

    # # Loading training data

    training_mat_set = ['set1.mat', 'set2.mat', 'set3.mat', 'set4.mat',
                        'set5.mat', 'set6.mat', 'set7.mat', 'set8.mat',
                        'set9.mat', 'set10.mat', 'set11.mat', 'set12.mat',
                        'set13.mat', 'set14.mat']

    base_path = os.getcwd()

    x = tf.placeholder(tf.float32, [None, VEC_LENGTH], name='x')
    max_nonzero = tf.placeholder(tf.int32, [], name='max_nonzero')

    W = ksparse_base.weight_variable([VEC_LENGTH, NUM_ATOMS])
    # W = ksparse_base.load_nnsc_dict()

    b = ksparse_base.bias_variable([NUM_ATOMS, 1])
    b_ = ksparse_base.bias_variable([VEC_LENGTH, 1])

    y, W = ksparse_base.ksparse_autoenc_withbias(x, W, b, b_, max_nonzero)

    # Defining L2 loss
    loss = tf.nn.l2_loss(x - y, name='reconstruction_error')
    tf.summary.scalar('L2 loss', loss)
    # Train using optimizer
    train_step = tf.train.AdamOptimizer(3e-3).minimize(loss)

    merged_summary = tf.summary.merge_all()

    # Learning

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    # Adding summary writers for Tensorflow
    token = now.strftime("%Y%m%d-%H%M%S") + "/"

    train_writer = tf.summary.FileWriter('train/' + token, sess.graph)
    test_writer = tf.summary.FileWriter('test/' + token, sess.graph)

    # Save initialization of W
    w_np = {'U': sess.run(W)}
    # Storing w as mat file
    dict_img = ksparse_base.create_image(w_np['U'], IM_EDGE, IM_CHANNELS)
    imsave(save_path_prefix + 'dict_init.png', dict_img)

    NUM_NONZERO = 70
    summary, loss_val = sess.run([merged_summary, loss], feed_dict={
        x: mnist.test.images, max_nonzero: NUM_NONZERO})
    print(loss_val)

    test_writer.add_summary(summary, 0)

    """
    Run
    """

    summ_step = 1

    NUM_NONZERO = 70
    # Loop over all mat files
    for i in range(3000):
        print("Train #", i)
        batch_xs, _ = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={x: batch_xs, max_nonzero: NUM_NONZERO})

        summary, loss_val = sess.run([merged_summary, loss],
                                     feed_dict={x: mnist.test.images,
                                                max_nonzero: NUM_NONZERO})

        print(loss_val)

        test_writer.add_summary(summary, summ_step)
        summ_step = summ_step + 1

    print("Optimization Finished!")
    saver.save(sess, save_path_prefix + "./trainedModel.ckpt")

    w_np = {'U': sess.run(W)}

    sess.close()

    # Storing w as mat file
    savemat(save_path_prefix + 'dict.mat', w_np)

    dict_img = ksparse_base.create_image(w_np['U'], IM_EDGE, IM_CHANNELS)
    imsave(save_path_prefix + 'dict.png', dict_img)
