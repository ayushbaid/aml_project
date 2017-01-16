# Train ksparse-autoencoder for patches

import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
from scipy.misc import imsave, imresize
import os
import math
from make_sparse_op import make_sparse


def create_image(patches, patch_size):
    # To display patches in a collage
    patches = np.squeeze(patches)
    num_patches = np.shape(patches)[1]

    im_size = patch_size * math.floor(math.sqrt(num_patches))

    # Performing contrast stretching for each patch
    min_vals = np.amin(patches, axis=0)
    max_vals = np.amax(patches, axis=0) + 1e-5

    patches = np.divide(patches - min_vals, max_vals - min_vals)

    img = np.zeros((im_size, im_size, 3))

    num_blocks = im_size // patch_size

    for row_idx in range(0, im_size, patch_size):
        for col_idx in range(0, im_size, patch_size):
            patch_idx = (row_idx // patch_size) * num_blocks + \
                        (col_idx // patch_size)

            img[row_idx:row_idx + patch_size, col_idx:col_idx +
                patch_size, :] = np.reshape(patches[:, patch_idx],
                                            (patch_size, patch_size, 3), order='F')

    return imresize(img, 10.0, 'nearest')


# Helper functions

def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.2)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def weight_init_prev():
    loaded_data = loadmat('ar_patch7_atoms1024-dict.mat')
    return tf.Variable(loaded_data['U'], dtype=tf.float32)


# def get_sparse_mask(x, k):
#     """
#     Create a mask to make x sparse along the last dimension.
#     x is a numpy array
#     """

#     # Get the indices of the top k elements in linear time
#     indices = np.argpartition(x, -k)[-k:]
#     mask = np.zeros(np.shape(x), dtype=bool)

#     mask[indices] = True

#     return mask


# def make_sparse(h, k, sess):
#     h_npy = np.tranpose(h.eval(session=sess))
#     k_int = k.eval(session=sess)
#     mask = tf.constant(get_sparse_mask(h_npy, k_int), dtype=tf.boolean)

#     h_sparse = tf.mul(h, tf.to_float(tf.transpose(mask)))

#     return h_sparse


def ksparse_autoenc(x, W, b, b_, NUM_NONZERO):
    # W = tf.Print(W, [W], "W: ")

    # Normalizing W
    # W = tf.nn.l2_normalize(W, 0, name='weight_normalize')

    h = tf.add(tf.matmul(W, x, transpose_a=True, transpose_b=True,
                         name='mult_encoder'), b, name='add_encoder')

    '''
    Select max elements
    '''
    h_sparse = tf.transpose(make_sparse(tf.transpose(h), NUM_NONZERO))

    """
    Reconstruction
    """
    y = tf.add(tf.matmul(W, h_sparse, name='mult_decoder'), b_, name='add_decoder')

    return tf.transpose(y)


if __name__ == '__main__':

    # Defining fixed params
    IM_EDGE = 7
    IM_CHANNELS = 3
    VEC_LENGTH = (IM_EDGE**2) * IM_CHANNELS

    NUM_ATOMS = 50

    TRAIN_BATCH_SIZE = 1000

    save_path_prefix = 'normal_eig_init/'

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

    W = weight_variable([VEC_LENGTH, NUM_ATOMS])
    # W = weight_init_prev()

    b = bias_variable([NUM_ATOMS, 1])
    b_ = bias_variable([VEC_LENGTH, 1])

    y = ksparse_autoenc(x, W, b, b_, max_nonzero)

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


    # Save initialization of W
    w_np = {'U': sess.run(W)}
    # Storing w as mat file
    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_init.png', dict_img)

    NUM_NONZERO = 10

    summary, _ = sess.run([merged_summary, loss], feed_dict={x: test_npy, max_nonzero: NUM_NONZERO})

    test_writer.add_summary(summary, 0)

    """
    Run
    """

    # summ_step = 1

    # # Loop over all mat files
    # for dataset in training_mat_set:
    #     print("%s Dataset loaded", dataset)
    #     loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

    #     input_data = loaded_data['training_set']
    #     loaded_data = None

    #     start_idx = 1
    #     num_items = TRAIN_BATCH_SIZE
    #     end_idx = start_idx + num_items

    #     while(end_idx < input_data.shape[1]):
    #         data = np.transpose(input_data[:, start_idx: end_idx])
    #         sess.run(train_step, feed_dict={x: data, max_nonzero: NUM_NONZERO})

    #         start_idx = end_idx
    #         end_idx = start_idx + num_items

    #         eval = sess.run(loss, feed_dict={x: test_npy,
    #                                          max_nonzero: NUM_NONZERO})

    #         test_writer.add_summary(eval, summ_step)
    #         summ_step = summ_step + 1

    # print("Optimization Finished!")
    # saver.save(sess, save_path_prefix + "./trainedModel.ckpt")

    w_np = {'U': sess.run(W)}

    sess.close()

    # Storing w as mat file
    savemat(save_path_prefix + 'dict.mat', w_np)

    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict.png', dict_img)
