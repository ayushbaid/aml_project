
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

def ksparse_autoenc(x, W, NUM_NONZERO, NUM_ATOMS, IM_SIZE):
    # W = tf.Print(W, [W], "W: ")
    W_temp = tf.transpose(tf.matrix_inverse(tf.transpose(W, [2,3,0,1])), [2,3,0,1])
    h = tf.nn.conv2d(x, W_temp, strides=[1, 1, 1, 1], padding='SAME')

    '''
    Select max elements
    '''
    h_sparse = tf.reshape( hard_thresholding(tf.reshape(h, [-1, NUM_ATOMS]) 
                    , NUM_NONZERO), [-1, IM_SIZE, IM_SIZE, NUM_ATOMS])
    """
    Reconstruction
    """
    y = tf.nn.depthwise_conv2d(h_sparse, tf.transpose(W, [1, 0, 3, 2]), 
                strides=[1, 1, 1, 1], padding='SAME')
    return tf.reduce_sum(y, [3], keep_dims = True)



if __name__ == '__main__':

    # Defining fixed params
    NUM_IMAGES = 24
    IM_SIZE = 150
    PATCH_SIZE = 7
    IM_CHANNELS = 3

    NUM_ATOMS = 1024
    

    TRAIN_BATCH_SIZE = 1e3




    base_path = os.getcwd()


    x = tf.placeholder(tf.float32, [NUM_IMAGES, IM_SIZE, IM_SIZE, 1])
    max_nonzero = tf.placeholder(tf.int32, [])

    W = weight_variable([PATCH_SIZE, PATCH_SIZE, 1, NUM_ATOMS])
    # W = weight_init_prev()

    y = ksparse_autoenc(x, W, max_nonzero, NUM_ATOMS, IM_SIZE)
    loss = tf.nn.l2_loss(x-y)

    train_step = tf.train.AdagradOptimizer(0.1).minimize(loss)


    # # Learning


    init = tf.initialize_all_variables()
    saver = tf.train.Saver()



    sess = tf.Session()
    sess.run(init)

    #### LEARNING FOR RED


    """
    Run with lower sparsity
    """

    NUM_NONZERO = 100


    loaded_data = loadmat(os.path.join(base_path, 'train_images/concat_r.mat'))

    input_data = np.expand_dims(loaded_data['concat_r'],3)
    loaded_data = None

    sess.run(train_step, feed_dict=
                {x:input_data, max_nonzero:NUM_NONZERO})

    print("Low Sparsity Red Optimization Finished!")
    saver.save(sess,"./cnn/trainedModelNorm_red_lowsparse.ckpt")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat('cnn/dict_red_low.mat', w_np)



    """
    Run with medium sparsity
    """

    NUM_NONZERO = 60


    loaded_data = loadmat(os.path.join(base_path, 'train_images/concat_r.mat'))

    input_data = np.expand_dims(loaded_data['concat_r'],3)
    loaded_data = None

    sess.run(train_step, feed_dict=
                {x:input_data, max_nonzero:NUM_NONZERO})

    print("Medium Sparsity Red Optimization Finished!")
    saver.save(sess,"./cnn/trainedModelNorm_red_medsparse.ckpt")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat('cnn/dict_red_med.mat', w_np)


    """
    Run with higher sparsity
    """
    NUM_NONZERO = 15


    loaded_data = loadmat(os.path.join(base_path, 'train_images/concat_r.mat'))

    input_data = np.expand_dims(loaded_data['concat_r'],3)
    loaded_data = None

    sess.run(train_step, feed_dict=
                {x:input_data, max_nonzero:NUM_NONZERO})

    print("High Sparsity Red Optimization Finished!")
    saver.save(sess,"./cnn/trainedModelNorm_red_highsparse.ckpt")

    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    savemat('cnn/dict_red_high.mat', w_np)



    sess.close()
