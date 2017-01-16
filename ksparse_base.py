"""Base file for ksparse-autoencoder."""

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
from scipy.misc import imresize
import math
from make_sparse_op import make_sparse


def create_image(patches, patch_size):
    """To display patches in a collage."""
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
                                            (patch_size, patch_size, 3),
                                            order='F')

    return imresize(img, 10.0, 'nearest')

# Helper functions


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0, stddev=0.5)
    return tf.Variable(initial)


def load_nnsc_dict():
    loaded = loadmat('ar_patch7_atoms1024-dict.mat')
    return tf.Variable(loaded['U'], dtype=tf.float32)


def bias_variable(shape):
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def ksparse_autoenc_withbias(x, W, b, b_, NUM_NONZERO):
    """K-sparse Autoencoder with bias in linear layers."""
    h = tf.add(tf.matmul(W, x, transpose_a=True, transpose_b=True,
                         name='mult_encoder'), b, name='add_encoder')

    # Make h sparse
    h_sparse = tf.transpose(make_sparse(tf.transpose(h), NUM_NONZERO))

    # Reconstruction
    y = tf.add(tf.matmul(W, h_sparse, name='mult_decoder'),
               b_, name='add_decoder')

    return tf.transpose(y)


def ksparse_autoenc_withoutbias(x, W, NUM_NONZERO):
    """K-sparse Autoencoder without bias in linear layers."""
    h = tf.matmul(W, x, transpose_a=True,
                  transpose_b=True, name='mult_encoder')

    # Make h sparse
    h_sparse = tf.transpose(make_sparse(tf.transpose(h), NUM_NONZERO))

    # Reconstruction
    y = tf.matmul(W, h_sparse, name='mult_decoder')

    return tf.transpose(y)
