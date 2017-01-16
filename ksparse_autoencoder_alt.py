import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
from scipy.misc import imsave, imresize
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

    return imresize(img, 10.0, 'nearest')


### Helper functions

def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.2)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)

def weight_init_prev():
    loaded_data = loadmat('7-eig.mat')
    return tf.Variable(loaded_data['V'], dtype = tf.float32)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape)
    return tf.Variable(initial)

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

    inds = tf.unpack(indices, axis=0, num=None)

    for i, idx in enumerate(inds):
        out.append(tf.sparse_tensor_to_dense(tf.SparseTensor(tf.reshape(\
            tf.cast(idx,tf.int64),[-1,1]), tf.tile([True], [tf.size(idx)]), \
            [d]), default_value = False, validate_indices=False))

    mask = tf.reshape(tf.pack(out, axis=0), inp_shape)

    return mask


def make_sparse(h, k):
    mask = tf.transpose(get_sparse_mask(tf.transpose(h), k))

    h_sparse = tf.mul(h, tf.to_float(mask))

    return h_sparse



def ksparse_autoenc(x, W, NUM_NONZERO):
    # W = tf.Print(W, [W], "W: ")

    # Normalizing W
    W = tf.nn.l2_normalize(W, 0, name='weight_normalize')

    h = tf.matmul(tf.transpose(W), x, name='mult_1')  

    '''
    Select max elements
    '''

    h_sparse = make_sparse(h, NUM_NONZERO)

    '''
    Reconstruction
    '''
    y = tf.matmul(W, h_sparse, name='mult_2')
    

    return y, W


if __name__ == '__main__':

    # Defining fixed params
    IM_EDGE = 7
    IM_CHANNELS = 3
    VEC_LENGTH = (IM_EDGE**2)*IM_CHANNELS

    NUM_ATOMS = 147
    

    TRAIN_BATCH_SIZE = 1000

    save_path_prefix = 'normal_eig_init/'


    # # Loading training data


    training_mat_set = ['set1.mat', 'set2.mat', 'set3.mat', 'set4.mat', 
                           'set5.mat', 'set6.mat', 'set7.mat', 'set8.mat',
                           'set9.mat', 'set10.mat', 'set11.mat', 'set12.mat',
                           'set13.mat', 'set14.mat']

    base_path = os.getcwd()

    # Load the training set
    test_npy = np.transpose(np.load(os.path.join(base_path, \
                    'test/test_set.npy')))
    test_npy = test_npy[:,1:5000]
    TEST_BATCH_SIZE = test_npy.shape[1]


    x_train = tf.placeholder(tf.float32, [VEC_LENGTH, TRAIN_BATCH_SIZE], \
                name='x_train')
    x_test = tf.placeholder(tf.float32, [VEC_LENGTH, TEST_BATCH_SIZE], \
                name='x_test')
    max_nonzero = tf.placeholder(tf.int32, [], name='max_nonzero')

    # W = weight_variable([VEC_LENGTH, NUM_ATOMS])
    W = weight_init_prev()

    y_train = ksparse_autoenc(x_train, W, max_nonzero)
    y_test = ksparse_autoenc(x_test, W, max_nonzero)

    # Defining L2 loss
    loss_train = tf.nn.l2_loss(x_train-y_train)
    loss_test = tf.nn.l2_loss(x_test-y_test)

    # Train using Adagrad
    train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(loss_train)

    merged_summary = tf.summary.merge_all()

    # Adding summary writers for Tensorflow
    train_writer = tf.summary.FileWriter('train', sess.graph)
    test_writer = tf.summary.FileWriter('test')


    ### Learning

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    sess = tf.Session()
    sess.run(init)

    # Save initialization of W
    w_np = {'U':sess.run(W)}
    # Storing w as mat file
    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict_init.png', dict_img)


    eval = sess.run(loss_test, feed_dict={x_test: test_npy, max_nonzero: \
                    NUM_NONZERO})

    test_writer.add_summary(eval, 0)



    """
    Run
    """
    NUM_NONZERO = 10

    summ_step = 1

    # Loop over all mat files
    for dataset in training_mat_set:
        print("%s Dataset loaded", dataset)
        loaded_data = loadmat(os.path.join(base_path, 'train', dataset))

        input_data = loaded_data['training_set']
        loaded_data = None

        start_idx = 1
        num_items = TRAIN_BATCH_SIZE
        end_idx = start_idx+num_items

        while(end_idx<input_data.shape[1]):
            sess.run(train_step, feed_dict=
                    {x_train:input_data[:, start_idx:end_idx], 
                        max_nonzero:NUM_NONZERO})

            start_idx = end_idx
            end_idx = start_idx+num_items

            eval = sess.run(loss_test, feed_dict={x_test:\
                        test_npy, max_nonzero: \
                        NUM_NONZERO})

            test_writer.add_summary(eval, summ_step)
            summ_step = summ_step + 1
            
    print("Optimization Finished!")
    saver.save(sess,save_path_prefix + "./trainedModelNorm.ckpt")

    w_np = {'U':sess.run(W)}

    sess.close()


    # Storing w as mat file
    savemat(save_path_prefix + 'dict.mat', w_np)

    dict_img = create_image(w_np['U'], IM_EDGE)
    imsave(save_path_prefix + 'dict.png', dict_img)

    
