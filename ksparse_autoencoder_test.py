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
    return l2_regularizer(tf.Variable(initial))

def bias_variable(shape):
    initial = tf.zeros(shape)
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

def l2_regularizer(W):
    """
    Performs L2 regularization along columns
    """

    l2_norm = tf.reduce_sum(tf.square(W), axis = 0, keep_dims = True)
    W = tf.truediv(W, l2_norm)

    return W

def get_sparse_mask(x, k):
    '''
    Create a mask to make x sparse along the last dimension
    '''

    inp_shape = tf.shape(x) # Storing the shape for mask creation
    d = x.get_shape().as_list()[-1] # Get the last dimension
    matrix_in = tf.reshape(x, [-1,d]) # Reshape into 2d matrix


    # Get indices across last dimension
    _ , indices = tf.nn.top_k(matrix_in, k=k, sorted=False)

    '''
    Creating a boolean mask using the indices
    '''
    out = []

    inds = tf.unpack(indices, axis=0)
    for i, idx in enumerate(inds):
        out.append(tf.sparse_tensor_to_dense(tf.SparseTensor(tf.reshape(\
            tf.cast(idx,tf.int64),[-1,1]), [True], [d]), \
            default_value = False, validate_indices=False))
    mask = tf.reshape(tf.pack(out), inp_shape)

    return mask


def make_sparse(h, k):
    mask = tf.transpose(get_sparse_mask(tf.transpose(h), k))

    h_sparse = tf.mul(h, tf.to_float(mask))

    return h_sparse



def ksparse_autoenc(x, W, b, b_, NUM_NONZERO, sess):
    # W = tf.Print(W, [W], "W: ")

    # Normalizing W
    W = tf.nn.l2_normalize(W, 0, name='weight_normalize')

    h = tf.add(tf.matmul(tf.transpose(W), x, name='mult_1'), b, name='add_1')  

    '''
    Select max elements
    '''

    h_sparse = make_sparse(h, NUM_NONZERO)

    """
    Reconstruction
    """
    y = tf.add(tf.matmul(W, h_sparse, name='mult_2'), b_, name='add_2')

    return y


if __name__ == '__main__':

    # Defining fixed params
    VEC_LENGTH = 6
    NUM_ATOMS = 3
    TRAIN_BATCH_SIZE = 2

    x = tf.placeholder(tf.float32, [VEC_LENGTH, TRAIN_BATCH_SIZE], name = 'x')
    max_nonzero = tf.placeholder(tf.int32, [], name='max_nonzero')

    sess = tf.Session()
    

    W = weight_variable([VEC_LENGTH, NUM_ATOMS])

    # W = weight_init_prev()
    b = bias_variable([NUM_ATOMS, TRAIN_BATCH_SIZE])
    b_ = bias_variable([VEC_LENGTH, TRAIN_BATCH_SIZE])

    ### Learning

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    sess.run(init)

    y = ksparse_autoenc(x, W, b, b_, max_nonzero, sess)

    # Defining L2 loss
    loss = tf.nn.l2_loss(x-y)

    # Train using Adagrad
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


    # Adding summary writers for Tensorflow
    train_writer = tf.summary.FileWriter('train', sess.graph)
    test_writer = tf.summary.FileWriter('test')



    NUM_NONZERO = 1


    for step in range(2000):
        summ,_ = sess.run([merged_summary, train_step], feed_dict=
                {x:np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[0,1]]), \
                        max_nonzero:NUM_NONZERO})
        train_writer.add_summary(summ, step)


    # Get the reconstructions
    x_ = y.eval(session = sess, feed_dict = {x:np.array([[1,0],[1,0],[1,0], \
                [0,1],[0,1],[0,1]]), max_nonzero:NUM_NONZERO})
    print('Reconstruction')
    print(x_)

    print('Output')
    print('W:')
    print(sess.run(W))
    print('b:')
    print(sess.run(b))
    print('b_:')
    print(sess.run(b_))

    sess.close()


    
