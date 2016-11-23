
# coding: utf-8

# In[3]:

import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
import os


# # Helper functions

# In[6]:

def weight_variable(shape):
    initial = tf.random_uniform(shape, minval = 0, maxval = 1)
    return tf.Variable(initial)

def weight_init_prev():
    loaded_data = loadmat('ar_patch7_atoms1024-dict.mat')


    return tf.transpose(tf.Variable(loaded_data['U'], dtype = tf.float32))

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

def pinv(A):
    return tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(A), A)), tf.transpose(A))

def ksparse_autoenc(x, W, NUM_NONZERO):
    # W = tf.Print(W, [W], "W: ")
    h = tf.matmul(x, pinv(W))

    '''
    Select max elements
    '''
    h_sparse = hard_thresholding(h, NUM_NONZERO)

    """
    Reconstruction
    """
    y = tf.matmul(h_sparse, W)

    return y



if __name__ == '__main__':

    # Defining fixed params
    IM_EDGE = 7
    IM_CHANNELS = 3
    VEC_LENGTH = (IM_EDGE**2)*IM_CHANNELS

    NUM_ATOMS = 1024
    

    TRAIN_BATCH_SIZE = 1e3


    # # Loading training data


    training_mat_set = ['set1.mat', 'set2.mat', 'set3.mat', 'set4.mat', 
                           'set5.mat', 'set6.mat', 'set7.mat', 'set8.mat',
                           'set9.mat', 'set10.mat', 'set11.mat', 'set12.mat',
                           'set13.mat', 'set14.mat']
    test_mat = 'test_set.mat'

    base_path = os.getcwd()


    x = tf.placeholder(tf.float32, [TRAIN_BATCH_SIZE, VEC_LENGTH])
    max_nonzero = tf.placeholder(tf.int32, [])

    # W = weight_variable([NUM_ATOMS, VEC_LENGTH])
    W = weight_init_prev()

    y = ksparse_autoenc(x, W, max_nonzero)
    loss = tf.nn.l2_loss(x-y)

    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)


    # # Learning


    init = tf.initialize_all_variables()
    saver = tf.train.Saver()



    sess = tf.Session()
    sess.run(init)


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
            sess.run(train_step, feed_dict=
                    {x:np.transpose(input_data[:, start_idx:end_idx]), 
                        max_nonzero:NUM_NONZERO})
            start_idx = end_idx
            end_idx = start_idx+num_items
            
    print("Low Sparsity Optimization Finished!")
    saver.save(sess,"./trainedModelNorm_lowsparse.ckpt")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat('dict_low.mat', w_np)



    """
    Run with medium sparsity
    """

    NUM_NONZERO = 60

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
            sess.run(train_step, feed_dict=
                    {x:np.transpose(input_data[:, start_idx:end_idx]), 
                        max_nonzero:NUM_NONZERO})
            start_idx = end_idx
            end_idx = start_idx+num_items
            
    print("Medium Sparsity Optimization Finished!")
    saver.save(sess,"./trainedModelNorm_medsparse.ckpt")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat('dict_med.mat', w_np)

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
            sess.run(train_step, feed_dict=
                    {x:np.transpose(input_data[:, start_idx:end_idx]), 
                        max_nonzero:NUM_NONZERO})
            start_idx = end_idx
            end_idx = start_idx+num_items
            
    print("High Sparsity Optimization Finished!")
    saver.save(sess,"./trainedModelNorm_highsparse.ckpt")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat('dict_high.mat', w_np)



    sess.close()
