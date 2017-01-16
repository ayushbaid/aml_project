import tensorflow as tf
import numpy as np    

if __name__=='__main__':

    sess = tf.InteractiveSession()

    x = tf.random_uniform([10, 4, 4, 3], minval=0, maxval=1, dtype=tf.float32)

    inp_shape = tf.shape(x)
    d = x.get_shape().as_list()[-1]
    matrix_in = tf.reshape(x, [-1,d])

    print(np.shape(x.eval()))
    print(np.shape(matrix_in.eval()))
    

    values, indices = tf.nn.top_k(matrix_in, k=2, sorted=False)
    out = []


    vals = tf.unpack(values, axis=0)
    inds = tf.unpack(indices, axis=0)
    for i, idx in enumerate(inds):
        out.append(tf.sparse_tensor_to_dense(tf.SparseTensor(tf.reshape(\
            tf.cast(idx,tf.int64),[-1,1]), [True], [d]), \
            default_value = False, validate_indices=False))
    mask = tf.reshape(tf.pack(out), inp_shape)

    sess.close()