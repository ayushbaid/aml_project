import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
from scipy.misc import imsave
import os
import math

### To display patches in a collage

def create_image(patches, patch_size):
    num_patches = np.shape(patches)[1]

    im_size = patch_size*math.floor(math.sqrt(num_patches))

    # Performing contrast stretching for each patch
    min_vals = np.amin(patches, axis=0)
    max_vals = np.amax(patches, axis=0)
    patches = np.divide(patches-min_vals, max_vals-min_vals)

    img = np.zeros((im_size, im_size, 3))

    num_blocks = im_size//patch_size

    for row_idx in range(0, im_size, patch_size):
        for col_idx in range(0, im_size, patch_size):
            patch_idx = (row_idx//patch_size)*num_blocks + \
                        (col_idx//patch_size)

            img[row_idx:row_idx+patch_size, col_idx:col_idx+patch_size,:] = \
                    np.reshape(patches[:,patch_idx], (patch_size, patch_size, \
                        3), order='F')

    return img


### Helper functions

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval = 0, maxval = 1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.random_uniform(shape)
    return tf.Variable(initial)

def weight_init_prev():
    loaded_data = loadmat('ar_patch7_atoms1024-dict.mat')
    return tf.Variable(loaded_data['U'], dtype = tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def hard_thresholding(x,k):
    values, indices = tf.nn.top_k(x, k)  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
    # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
    my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)  # will be [[0], [1]]
    my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]
    
    # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
    full_indices = tf.concat(2, [tf.expand_dims(my_range_repeated, 2), tf.expand_dims(indices, 2)]) 
    full_indices = tf.reshape(full_indices, [-1, 2])
    x_sparse = tf.sparse_to_dense(full_indices, x.get_shape(), 
                        tf.reshape(values, [-1]), default_value = 0., 
                          validate_indices = False)
    return x_sparse



def ksparse_autoenc(x, W, b, b_, NUM_NONZERO):
    # W = tf.Print(W, [W], "W: ")
    h = tf.add(tf.matmul(tf.transpose(W), x), b)  

    '''
    Select max elements
    '''
    h_sparse = hard_thresholding(h, NUM_NONZERO)

    """
    Reconstruction
    """
    y = tf.add(tf.matmul(W, h_sparse), b_)

    return y


if __name__ == '__main__':

    # Defining fixed params
    IM_EDGE = 7
    IM_CHANNELS = 3
    VEC_LENGTH = (IM_EDGE**2)*IM_CHANNELS

    NUM_ATOMS = 1024
    

    TRAIN_BATCH_SIZE = 1e3

    save_path_prefix = 'normal_nnsc_init/'


    # # Loading training data


    training_mat_set = ['set1.mat', 'set2.mat', 'set3.mat', 'set4.mat', 
                           'set5.mat', 'set6.mat', 'set7.mat', 'set8.mat',
                           'set9.mat', 'set10.mat', 'set11.mat', 'set12.mat',
                           'set13.mat', 'set14.mat']
    test_mat = 'test_set.mat'

    base_path = os.getcwd()


    x = tf.placeholder(tf.float32, [VEC_LENGTH, TRAIN_BATCH_SIZE])
    max_nonzero = tf.placeholder(tf.int32, [])

    # W = weight_variable([VEC_LENGTH, NUM_ATOMS])
    W = weight_init_prev()
    b = bias_variable([NUM_ATOMS, TRAIN_BATCH_SIZE])
    b_ = bias_variable([VEC_LENGTH, TRAIN_BATCH_SIZE])

    y = ksparse_autoenc(x, W, b, b_, max_nonzero)

    # Defining L2 loss
    loss = tf.nn.l2_loss(x-y)

    # Train using Adagrad
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    ### Learning

    init = tf.global_variables_initializer()
    init1 = tf.local_variables_initializer()
    saver = tf.train.Saver()


    sess = tf.Session()
    sess.run(init)
    sess.run(init1)

    # Save initialization of W
    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_init.png', dict_img)


    """
    Run with lower sparsity
    """

    NUM_NONZERO = 500

    # Loop over all mat files
    for dataset in training_mat_set:
        print("Dataset loaded")
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx+num_items
        while(end_idx<input_data.shape[1]):
            for step in range(2000):
                sess.run(train_step, feed_dict=
                        {x:input_data[:, start_idx:end_idx], 
                            max_nonzero:NUM_NONZERO})
            start_idx = end_idx
            end_idx = start_idx+num_items
            
    print("Low Sparsity Optimization Finished!")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat(save_path_prefix + 'dict_low.mat', w_np)
    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_low.png', dict_img)



    """
    Run with medium sparsity
    """

    NUM_NONZERO = 80

    # Loop over all mat files
    for dataset in training_mat_set:
        print("Dataset loaded")
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx+num_items
        while(end_idx<input_data.shape[1]):
            for step in range(2000):
                sess.run(train_step, feed_dict=
                        {x:input_data[:, start_idx:end_idx], 
                            max_nonzero:NUM_NONZERO})
            start_idx = end_idx
            end_idx = start_idx+num_items
            
    print("Medium Sparsity Optimization Finished!")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat(save_path_prefix + 'dict_med.mat', w_np)
    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_med.png', dict_img)

    """
    Run with higher sparsity
    """
    NUM_NONZERO = 15

    # Loop over all mat files
    for dataset in training_mat_set:
        print("Dataset loaded")
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx+num_items
        while(end_idx<input_data.shape[1]):
            for step in range(2000):
                sess.run(train_step, feed_dict=
                        {x:input_data[:, start_idx:end_idx], 
                            max_nonzero:NUM_NONZERO})
            start_idx = end_idx
            end_idx = start_idx+num_items
            
    print("High Sparsity Optimization Finished!")
    saver.save(sess,save_path_prefix + "./trainedModelNorm_highsparse.ckpt")

    w_np = {'U':sess.run(W)}

    sess.close()


    # Storing w as mat file
    savemat(save_path_prefix + 'dict_high.mat', w_np)

    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_high.png', dict_img)

    
