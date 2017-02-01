# Train ksparse-autoencoder for patches

import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
from scipy.misc import imsave
import ksparse_base
import os


if __name__ == '__main__':

    # Defining fixed params
    IM_EDGE = 7
    IM_CHANNELS = 3
    VEC_LENGTH = (IM_EDGE**2) * IM_CHANNELS

    NUM_ATOMS = 100
    TRAIN_BATCH_SIZE = 200

    save_path_prefix = 'normal_nnsc_init/'

    # # Loading training data

    training_mat_set = ['set1.mat', 'set2.mat', 'set3.mat', 'set4.mat',
                        'set5.mat', 'set6.mat', 'set7.mat', 'set8.mat',
                        'set9.mat', 'set10.mat', 'set11.mat', 'set12.mat',
                        'set13.mat', 'set14.mat']

    base_path = os.getcwd()

    # Load the training set
    test_npy = np.load(os.path.join(base_path, 'test/test_set.npy'))
    test_npy = test_npy[1:5000, :]
    TEST_BATCH_SIZE = test_npy.shape[1]

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
    train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(loss)

    merged_summary = tf.summary.merge_all()

    # Learning

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)

    # Adding summary writers for Tensorflow
    train_writer = tf.summary.FileWriter('train', sess.graph)
    test_writer = tf.summary.FileWriter('test', sess.graph)

    # Save initialization of W
    w_np = {'U': sess.run(W)}
    # Storing w as mat file
    dict_img = ksparse_base.create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_init.png', dict_img)

    NUM_NONZERO = 10
    summary, _ = sess.run([merged_summary, loss], feed_dict={
                          x: test_npy, max_nonzero: NUM_NONZERO})

    test_writer.add_summary(summary, 0)

    """
    Run
    """

    summ_step = 1

    NUM_NONZERO = 100
    # Loop over all mat files
    for dataset in training_mat_set:
        print("Dataset loaded: ", dataset)
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx + num_items

        while(end_idx < input_data.shape[1]):
            data = np.transpose(input_data[:, start_idx: end_idx])
            sess.run(train_step, feed_dict={x: data, max_nonzero: NUM_NONZERO})

            start_idx = end_idx
            end_idx = start_idx + num_items

            summary, _ = sess.run([merged_summary, loss],
                                  feed_dict={x: test_npy,
                                             max_nonzero: NUM_NONZERO})

            test_writer.add_summary(summary, summ_step)
            summ_step = summ_step + 1

    NUM_NONZERO = 40
    # Loop over all mat files
    for dataset in training_mat_set:
        print("Dataset loaded: ", dataset)
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx + num_items

        while(end_idx < input_data.shape[1]):
            data = np.transpose(input_data[:, start_idx: end_idx])
            sess.run(train_step, feed_dict={x: data, max_nonzero: NUM_NONZERO})

            start_idx = end_idx
            end_idx = start_idx + num_items

            summary, _ = sess.run([merged_summary, loss],
                                  feed_dict={x: test_npy,
                                             max_nonzero: NUM_NONZERO})

            test_writer.add_summary(summary, summ_step)
            summ_step = summ_step + 1

    NUM_NONZERO = 10
    # Loop over all mat files
    for dataset in training_mat_set:
        print("Dataset loaded: ", dataset)
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx + num_items

        while(end_idx < input_data.shape[1]):
            data = np.transpose(input_data[:, start_idx: end_idx])
            sess.run(train_step, feed_dict={x: data, max_nonzero: NUM_NONZERO})

            start_idx = end_idx
            end_idx = start_idx + num_items

            summary, _ = sess.run([merged_summary, loss],
                                  feed_dict={x: test_npy,
                                             max_nonzero: NUM_NONZERO})

            test_writer.add_summary(summary, summ_step)
            summ_step = summ_step + 1

    print("Optimization Finished!")
    saver.save(sess, save_path_prefix + "./trainedModel.ckpt")

    w_np = {'U': sess.run(W)}

    sess.close()

    # Storing w as mat file
    savemat(save_path_prefix + 'dict.mat', w_np)

    dict_img = ksparse_base.create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict.png', dict_img)
