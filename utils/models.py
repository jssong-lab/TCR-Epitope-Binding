import os
import keras

import pandas as pd
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt

from keras import backend as K
from keras import metrics
from keras.models import Sequential, model_from_json
from keras.layers import Convolution1D, Lambda
from keras.optimizers import Adam
from keras.layers import Input, MaxPool1D, Dropout, LSTM, Bidirectional
from keras.models import Model
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import UpSampling1D
from keras.layers import BatchNormalization
from keras.layers import Layer

TCR_PAD_LEN = 20
EP_PAD_LEN = 10
NCH = 6
TCR_SHAPE=(TCR_PAD_LEN, NCH)
EP_SHAPE = (EP_PAD_LEN, NCH)
n_dims = 32
BATCH_SIZE = 128

def make_base_network(input_shape):
    model = Sequential()
    model.add(Convolution1D(filters=256, kernel_size=5, strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution1D(filters=128, kernel_size=5, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    model.summary()
    return model

def embed_TCR_network(input_shape, viz):
    model = Sequential()
    model.add(Convolution1D(filters=128, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution1D(filters=64, kernel_size=4, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(n_dims))
    if viz:
        model.summary()
    return model

def embed_TCR_network_archive(input_shape):
    model = Sequential()
    model.add(Convolution1D(filters=128, kernel_size=3, strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution1D(filters=64, kernel_size=4, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(n_dims))
    model.add(BatchNormalization())
    model.summary()
    return model

def embed_TCR_network_archive(input_shape):
    model = Sequential()
    model.add(Convolution1D(filters=128, kernel_size=4, strides=1, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(MaxPool1D(pool_size=3, strides=2, padding='same'))
    #model.add(Convolution1D(filters=64, kernel_size=3, strides=1, padding='same'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPool1D(pool_size=3, strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(n_dims))
    model.add(BatchNormalization())
    model.summary()
    return model

def make_ep_network(input_shape, viz):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Convolution1D(filters=64, kernel_size=5, strides=1, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Flatten()(x)
    z_mean = Dense(n_dims)(x)
    z_log_var = Dense(n_dims)(x)
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    if viz:
        encoder.summary()
    return encoder

def make_ep_network_archive_20190121(input_shape):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Convolution1D(filters=64, kernel_size=5, strides=1, padding='same')(inputs)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    z_mean = Dense(32)(x)
    z_mean = Activation('tanh')(z_mean)
    z_log_var = Dense(32)(x)
    z_log_var = Activation('tanh')(z_log_var)
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    return encoder

def make_ep_network_archive(input_shape):
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Convolution1D(filters=64, kernel_size=5, strides=1, padding='same')(inputs)
    x = Activation('tanh')(x)
    x = Flatten()(x)
    z_mean = Dense(32)(x)
    z_mean = Activation('tanh')(z_mean)
    z_log_var = Dense(32)(x)
    z_log_var = Activation('relu')(z_log_var)
    encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
    encoder.summary()
    return encoder

def get_dist_layer_cosine_sized(vects):
    x = vects[0]
    mean = vects[1]
    var = vects[2]
    
    dif = x - mean
    
    X = vects[0]
    y = vects[1]
    y = K.squeeze(y, axis=1)
    batch_size = set_size
    pairwise = _pairwise_distances_cosine(X)

    pos_mask_pre = K.tf.to_float(K.tf.equal(K.expand_dims(y, axis=1), K.expand_dims(y, axis=0)))
    pos_mask = pos_mask_pre - K.tf.eye(batch_size)
    dist_pos = K.tf.multiply(pairwise, pos_mask)

    neg_mask = K.tf.to_float(K.tf.not_equal(K.expand_dims(y, axis=1), K.expand_dims(y, axis=0)))
    dist_neg_tmp = K.tf.multiply(pairwise, neg_mask)

    intra = K.sum(dist_pos)/K.sum(pos_mask)
    inter = K.sum(dist_neg_tmp)/K.sum(neg_mask)
    return [intra, inter, intra/inter, dist_pos, dist_neg_tmp]


def semi_hard_layer_cosine(vects):
    X, y = vects[0], vects[1]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances_cosine(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = K.tf.cast(K.tf.eye(K.tf.shape(labels)[0]), K.tf.bool)
    indices_not_equal = K.tf.logical_not(indices_equal)
    i_not_equal_j = K.tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.tf.expand_dims(indices_not_equal, 0)

    distinct_indices = K.tf.logical_and(K.tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = K.tf.equal(K.tf.expand_dims(labels, 0), K.tf.expand_dims(labels, 1))
    i_equal_j = K.tf.expand_dims(label_equal, 2)
    i_equal_k = K.tf.expand_dims(label_equal, 1)

    valid_labels = K.tf.logical_and(i_equal_j, K.tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = K.tf.logical_and(distinct_indices, valid_labels)
    mask = K.tf.to_float(mask)
    
    return mask

def _get_triplet_mask_neg(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = K.tf.cast(K.tf.eye(K.tf.shape(labels)[0]), K.tf.bool)
    indices_not_equal = K.tf.logical_not(indices_equal)
    i_not_equal_j = K.tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = K.tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = K.tf.expand_dims(indices_not_equal, 0)

    distinct_indices = K.tf.logical_and(K.tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = K.tf.equal(K.tf.expand_dims(labels, 0), K.tf.expand_dims(labels, 1))
    
    nonneg = K.tf.logical_not(K.tf.equal(labels, -1))
    nonneg_mask = K.tf.logical_and(K.tf.expand_dims(nonneg, 0), K.tf.expand_dims(nonneg, 1))
    mod_equal = K.tf.logical_and(nonneg_mask, label_equal)
    i_equal_j = K.tf.expand_dims(mod_equal, 2)
    i_equal_k = K.tf.expand_dims(mod_equal, 1)
    valid_labels = K.tf.logical_and(i_equal_j, K.tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = K.tf.logical_and(distinct_indices, valid_labels)
    mask = K.tf.to_float(mask)
    
    return mask

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = K.tf.matmul(embeddings, K.tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = K.tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = K.tf.expand_dims(square_norm, 0) - 2.0 * dot_product + K.tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = K.tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = K.tf.to_float(K.tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = K.tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _pairwise_distances_cosine(embeddings):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """

    dot_product = K.tf.matmul(embeddings, K.tf.transpose(embeddings))


    square_norm = K.tf.diag_part(dot_product)

    den = K.tf.expand_dims(square_norm, 0) * K.tf.expand_dims(square_norm, 1)

    mask = K.tf.to_float(K.tf.equal(den, 0.0))
    den = den + mask * 1e-16
    den = K.tf.sqrt(den)
    den = den * (1.0 - mask)
    
    out = 0.5*(1 - dot_product/den)
    return out

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = K.tf.matmul(embeddings, K.tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = K.tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = K.tf.expand_dims(square_norm, 0) - 2.0 * dot_product + K.tf.expand_dims(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = K.tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = K.tf.to_float(K.tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = K.tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def gaussian_nll(ytrue, y_pred):
    """Keras implmementation og multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    n_dims=32
    mu = y_pred[0]
    logsigma = y_pred[1]
    x = ytrue
    
    mse = -0.5*K.sum(K.square((x-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return 0.03*K.mean(-log_likelihood)


def triplet_layer_archive(vects):
    X, y = vects[0], vects[3]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.3
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)    
    B=30
    A=0
    return B*triplet_loss



def triplet_layer(vects):
    X, y = vects[0], vects[1]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)  
    return triplet_loss

def triplet_layer_new(vects):
    X, y = vects[0], vects[1]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)  
    return triplet_loss

def triplet_layer_neg(vects):
    X, y = vects[0], vects[1]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask_neg(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)  
    return triplet_loss

def gaussian_nll_layer(vects):
    """Keras implmementation og multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    n_dims=32
    mu = vects[1]
    logsigma = vects[2]
    x = vects[0]
    
    mse = -0.5*K.sum(K.square((x-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi
    #kl = -K.sum(logsigma) + K.sum(K.exp(logsigma))

    return K.mean(-log_likelihood) #+ kl



def full_layer(vects):
    tri_out = vects[0]
    nll_out = vects[1]
    
    nll_val = K.mean(nll_out)
    A = 0.01
    B = 1
    return A*nll_val + B*tri_out

def gaussian_nll_layer_new(vects):
    """Keras implmementation og multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    n_dims=32
    mu = vects[1]
    logsigma = vects[2]
    x = vects[0]
    
    mse = -0.5*K.sum(K.square(x-mu)/K.exp(logsigma),axis=1)
    sigma_trace = -0.5*K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi
    #kl = -K.sum(logsigma) + K.sum(K.exp(logsigma))

    return -log_likelihood

def gaussian_nll_layer_new_v2_archive(vects):

    n_dims=32
    x = vects[0]
    mu1 = vects[1]
    logsigma1 = vects[2]
    mu2 = vects[3]
    logsigma2 = vects[4]
    mu3 = vects[5]
    logsigma3 = vects[6]
    
    
    mse1 = -0.5*K.sum(K.square(x-mu1)/K.exp(logsigma1),axis=1)
    sigma_trace1 = -0.5*K.sum(logsigma1, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood1 = mse1+sigma_trace1+log2pi
    
    mse2 = -0.5*K.sum(K.square(x-mu2)/K.exp(logsigma2),axis=1)
    sigma_trace2 = -0.5*K.sum(logsigma2, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood2 = mse2+sigma_trace2+log2pi
    
    mse3 = -0.5*K.sum(K.square(x-mu3)/K.exp(logsigma3),axis=1)
    sigma_trace3 = -0.5*K.sum(logsigma3, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood3 = mse3+sigma_trace3+log2pi
    
    #out = K.concatenate([log_likelihood1, log_likelihood2, log_likelihood3], axis=0)
    #out = K.transpose(log_likelihood1)
    out1 = K.tf.expand_dims(log_likelihood1, 1)
    out2 = K.tf.expand_dims(log_likelihood2, 1)
    out3 = K.tf.expand_dims(log_likelihood3, 1)
    out = K.concatenate([out1, out2, out3], axis=1)
    return out

def gaussian_nll_layer_new_v2(vects):

    n_dims=32
    x = vects[0]
    mu1 = vects[1]
    logsigma1 = vects[2]
    mu2 = vects[3]
    logsigma2 = vects[4]
    mu3 = vects[5]
    logsigma3 = vects[6]
    mu4 = vects[7]
    logsigma4 = vects[8]
    mu5 = vects[9]
    logsigma5 = vects[10]
    mu6 = vects[11]
    logsigma6 = vects[12]
    mu7 = vects[13]
    logsigma7 = vects[14]
    
    
    mse1 = -0.5*K.sum(K.square(x-mu1)/K.exp(logsigma1),axis=1)
    sigma_trace1 = -0.5*K.sum(logsigma1, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood1 = mse1+sigma_trace1+log2pi
    
    mse2 = -0.5*K.sum(K.square(x-mu2)/K.exp(logsigma2),axis=1)
    sigma_trace2 = -0.5*K.sum(logsigma2, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood2 = mse2+sigma_trace2+log2pi
    
    mse3 = -0.5*K.sum(K.square(x-mu3)/K.exp(logsigma3),axis=1)
    sigma_trace3 = -0.5*K.sum(logsigma3, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood3 = mse3+sigma_trace3+log2pi

    mse4 = -0.5*K.sum(K.square(x-mu4)/K.exp(logsigma4),axis=1)
    sigma_trace4 = -0.5*K.sum(logsigma4, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood4 = mse4+sigma_trace4+log2pi
    
    mse5 = -0.5*K.sum(K.square(x-mu5)/K.exp(logsigma5),axis=1)
    sigma_trace5 = -0.5*K.sum(logsigma5, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood5 = mse5+sigma_trace5+log2pi
    
    mse6 = -0.5*K.sum(K.square(x-mu6)/K.exp(logsigma6),axis=1)
    sigma_trace6 = -0.5*K.sum(logsigma6, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood6 = mse6+sigma_trace6+log2pi
    
    mse7 = -0.5*K.sum(K.square(x-mu7)/K.exp(logsigma7),axis=1)
    sigma_trace7 = -0.5*K.sum(logsigma7, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    log_likelihood7 = mse7+sigma_trace7+log2pi
    
    #out = K.concatenate([log_likelihood1, log_likelihood2, log_likelihood3], axis=0)
    #out = K.transpose(log_likelihood1)
    out1 = K.tf.expand_dims(log_likelihood1, 1)
    out2 = K.tf.expand_dims(log_likelihood2, 1)
    out3 = K.tf.expand_dims(log_likelihood3, 1)
    out4 = K.tf.expand_dims(log_likelihood4, 1)
    out5 = K.tf.expand_dims(log_likelihood5, 1)
    out6 = K.tf.expand_dims(log_likelihood6, 1)
    out7 = K.tf.expand_dims(log_likelihood7, 1)
    out = K.concatenate([out1, out2, out3, out4, out5, out6, out7], axis=1)
    out = K.softmax(out)
    return out

def experimental_layer(vects):
    
    n_dims=32
    mu = vects[1]
    logsigma = vects[2]
    x = vects[0]
    
    mse = -0.5*K.sum(K.square(x-mu)/K.exp(logsigma),axis=1)
    sigma_trace = -0.5*K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = (mse+sigma_trace+log2pi)

    X, y = vects[0], vects[3]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)    
    B=1
    A=0.0001
    return A*K.mean(-log_likelihood) + B*triplet_loss

def experimental_layer_neg(vects):
    
    n_dims=32
    mu = vects[1]
    logsigma = vects[2]
    x = vects[0]
    X, y = vects[0], vects[3]
    y = K.squeeze(y, axis=1)
    
    label_neg = K.tf.equal(y, -1)
    label_nonneg = K.tf.logical_not(label_neg)
    mask_ll = K.tf.to_float(label_nonneg)
    
    mse = -0.5*K.sum(K.square(x-mu)/K.exp(logsigma),axis=1)
    sigma_trace = -0.5*K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = (mse+sigma_trace+log2pi)

    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask_neg(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)    
    B=1
    A=0.0001
    log_likelihood_mask = K.tf.multiply(mask, log_likelihood)
    return A*K.mean(-log_likelihood_mask) + B*triplet_loss

def experimental_layer_no_class(vects):
    
    n_dims=32
    mu = vects[1]
    logsigma = vects[2]
    x = vects[0]
    
    mse = -0.5*K.sum(K.square(x-mu)/K.exp(logsigma),axis=1)
    sigma_trace = -0.5*K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = (mse+sigma_trace+log2pi)

    X, y = vects[0], vects[3]
    y = K.squeeze(y, axis=1)
    pairwise_dist = _pairwise_distances(X)
    
    anchor_positive_dist = K.tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = K.tf.expand_dims(pairwise_dist, 1)
    margin=0.1
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(y)
    mask = K.tf.to_float(mask)
    triplet_loss = K.tf.multiply(mask, triplet_loss)
    
    triplet_loss = K.tf.maximum(triplet_loss, 0.0)
    
    valid_triplets = K.tf.to_float(K.tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = K.tf.reduce_sum(valid_triplets)
    num_valid_triplets = K.tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = K.tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)    
    B=1
    A=0.005
    return A*K.mean(-log_likelihood) + B*triplet_loss

def identity_loss(y_true, y_pred):
    return y_pred

def make_tcr_gaussian_prior_layer(x):
    mse = -0.5*K.sum(K.square(x),axis=1)
    sigma_trace = -n_dims
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi
    return K.mean(-log_likelihood)

opt = Adam(1e-5)

def get_dist_layer_sized(vects):
    X = vects[0]
    y = vects[1]
    y = K.squeeze(y, axis=1)
    batch_size = BATCH_SIZE
    pairwise = _pairwise_distances(X)

    pos_mask_pre = K.tf.to_float(K.tf.equal(K.expand_dims(y, axis=1), K.expand_dims(y, axis=0)))
    pos_mask = pos_mask_pre - K.tf.eye(batch_size)
    dist_pos = K.tf.multiply(pairwise, pos_mask)
    
    neg_mask = K.tf.to_float(K.tf.not_equal(K.expand_dims(y, axis=1), K.expand_dims(y, axis=0)))
    dist_neg_tmp = K.tf.multiply(pairwise, neg_mask)

    intra = K.sum(dist_pos)/K.sum(pos_mask)
    inter = K.sum(dist_neg_tmp)/K.sum(neg_mask)
    return [intra, inter, intra/inter]

def make_full_network_new(viz):
    
    input_tcr = Input(shape=TCR_SHAPE)
    tcr_network = embed_TCR_network(TCR_SHAPE, viz)
    proc_tcr = tcr_network(input_tcr)
    
    input_ep = Input(shape=EP_SHAPE)
    ep_network = make_ep_network(EP_SHAPE, viz)
    proc_ep = ep_network(input_ep)
    
    y_class = Input(shape=(1,))
    
    dist_layer = Lambda(get_dist_layer_sized)([proc_tcr, y_class])
    dist_model = Model(inputs=[input_tcr, y_class], outputs=dist_layer)
    dist_model.compile(loss=identity_loss, optimizer=opt)
    
    nll_loss_layer = Lambda(gaussian_nll_layer_new)([proc_tcr, proc_ep[0], proc_ep[1]])
    nll_model = Model(inputs=[input_tcr, input_ep], outputs=nll_loss_layer)
    nll_out = nll_model([input_tcr, input_ep])
    nll_model.compile(loss=identity_loss, optimizer=opt)
    
    tcr_prior_layer = Lambda(make_tcr_gaussian_prior_layer)(proc_tcr)
    tcr_prior_model = Model(inputs=input_tcr, outputs=tcr_prior_layer)
    tcr_prior_model.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_triplet = Lambda(triplet_layer_new)([proc_tcr, y_class])
    triplet = Model(inputs=[input_tcr, y_class], outputs=loss_layer_triplet)
    triplet_out = triplet([input_tcr, y_class])
    triplet.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_full = Lambda(full_layer)([triplet_out, nll_out])
    full = Model(inputs=[input_tcr, input_ep, y_class], outputs=loss_layer_full)
    full.compile(loss=identity_loss, optimizer=opt)
    if viz:
        full.summary()
    
    experimental_loss_layer = Lambda(experimental_layer)([proc_tcr, proc_ep[0], proc_ep[1], y_class])
    exp = Model(inputs=[input_tcr, input_ep, y_class], outputs=experimental_loss_layer)
    exp.compile(loss=identity_loss, optimizer=opt)
    model_dict = {'Encoder': exp,
                  'Triple': triplet,
                  'NLL': nll_model,
                  'TCR_Embed': tcr_network,
                  'Ep_Embed': ep_network,
                  'TCR_Gaussian_Prior': tcr_prior_model,
                  'Dist': dist_model,
                  'Full': full}
    return model_dict

def make_full_network_neg(viz):
    
    input_tcr = Input(shape=TCR_SHAPE)
    tcr_network = embed_TCR_network(TCR_SHAPE, viz)
    proc_tcr = tcr_network(input_tcr)
    
    input_ep = Input(shape=EP_SHAPE)
    ep_network = make_ep_network(EP_SHAPE, viz)
    proc_ep = ep_network(input_ep)
    
    y_class = Input(shape=(1,))
    
    dist_layer = Lambda(get_dist_layer_sized)([proc_tcr, y_class])
    dist_model = Model(inputs=[input_tcr, y_class], outputs=dist_layer)
    dist_model.compile(loss=identity_loss, optimizer=opt)
    
    nll_loss_layer = Lambda(gaussian_nll_layer_new)([proc_tcr, proc_ep[0], proc_ep[1]])
    nll_model = Model(inputs=[input_tcr, input_ep], outputs=nll_loss_layer)
    nll_out = nll_model([input_tcr, input_ep])
    nll_model.compile(loss=identity_loss, optimizer=opt)
    
    tcr_prior_layer = Lambda(make_tcr_gaussian_prior_layer)(proc_tcr)
    tcr_prior_model = Model(inputs=input_tcr, outputs=tcr_prior_layer)
    tcr_prior_model.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_triplet = Lambda(triplet_layer_neg)([proc_tcr, y_class])
    triplet = Model(inputs=[input_tcr, y_class], outputs=loss_layer_triplet)
    triplet_out = triplet([input_tcr, y_class])
    triplet.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_full = Lambda(full_layer)([triplet_out, nll_out])
    full = Model(inputs=[input_tcr, input_ep, y_class], outputs=loss_layer_full)
    full.compile(loss=identity_loss, optimizer=opt)
    if viz:
        full.summary()
    
    experimental_loss_layer = Lambda(experimental_layer_neg)([proc_tcr, proc_ep[0], proc_ep[1], y_class])
    exp = Model(inputs=[input_tcr, input_ep, y_class], outputs=experimental_loss_layer)
    exp.compile(loss=identity_loss, optimizer=opt)
    model_dict = {'Encoder': exp,
                  'Triple': triplet,
                  'NLL': nll_model,
                  'TCR_Embed': tcr_network,
                  'Ep_Embed': ep_network,
                  'TCR_Gaussian_Prior': tcr_prior_model,
                  'Dist': dist_model,
                  'Full': full}
    return model_dict

def make_classifier(input_shape):
    model = Sequential()
    model.add(Convolution1D(filters=1, kernel_size=1, strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('softmax'))
    model.summary()
    return model

def make_full_network_new_v2():
    
    input_tcr = Input(shape=TCR_SHAPE)
    tcr_network = embed_TCR_network(TCR_SHAPE)
    proc_tcr = tcr_network(input_tcr)
    
    input_ep = Input(shape=EP_SHAPE)
    ep_network = make_ep_network(EP_SHAPE)
    proc_ep = ep_network(input_ep)
    
    inp_ep1 = Input(shape=EP_SHAPE)
    proc_ep1 = ep_network(inp_ep1)
    
    inp_ep2 = Input(shape=EP_SHAPE)
    proc_ep2 = ep_network(inp_ep2)
    
    inp_ep3 = Input(shape=EP_SHAPE)
    proc_ep3 = ep_network(inp_ep3)
    
    inp_ep4 = Input(shape=EP_SHAPE)
    proc_ep4 = ep_network(inp_ep4)
    
    inp_ep5 = Input(shape=EP_SHAPE)
    proc_ep5 = ep_network(inp_ep5)
    
    inp_ep6 = Input(shape=EP_SHAPE)
    proc_ep6 = ep_network(inp_ep6)
    
    inp_ep7 = Input(shape=EP_SHAPE)
    proc_ep7 = ep_network(inp_ep7)
    
    nll_loss_layer_v2 = Lambda(gaussian_nll_layer_new_v2)([proc_tcr, proc_ep1[0], proc_ep1[1], 
                                                        proc_ep2[0], proc_ep2[1], 
                                                        proc_ep3[0], proc_ep3[1],
                                                          proc_ep4[0], proc_ep4[1],
                                                          proc_ep5[0], proc_ep5[1],
                                                          proc_ep6[0], proc_ep6[1],
                                                          proc_ep7[0], proc_ep7[1]])
    nll_model_v2 = Model(inputs=[input_tcr, inp_ep1, inp_ep2, inp_ep3, inp_ep4, inp_ep5, inp_ep6, inp_ep7], outputs=nll_loss_layer_v2)
    nll_model_v2.compile(loss='categorical_crossentropy', optimizer=opt)
    #nll_out = nll_model([input_tcr, input_ep])
    #nll_out_v2 = nll_model_v2([input_tcr, inp_ep1, inp_ep2, inp_ep3])
    #input_nll = Input(shape=(3,))
    #class_out = nll_model_v2([input_tcr, inp_ep1, inp_ep2, inp_ep3])
    #class_out_net = make_classifier(input_nll)
    #proc_out = class_out_net(nll_out_v2)
    #class_out = nll_model_v2([input_tcr, inp_ep1, inp_ep2, inp_ep3])
    
    y_class = Input(shape=(1,))
    
    dist_layer = Lambda(get_dist_layer_sized)([proc_tcr, y_class])
    dist_model = Model(inputs=[input_tcr, y_class], outputs=dist_layer)
    dist_model.compile(loss=identity_loss, optimizer=opt)
    
    nll_loss_layer = Lambda(gaussian_nll_layer_new)([proc_tcr, proc_ep[0], proc_ep[1]])
    nll_model = Model(inputs=[input_tcr, input_ep], outputs=nll_loss_layer)
    nll_out = nll_model([input_tcr, input_ep])
    nll_model.compile(loss=identity_loss, optimizer=opt)
    
    tcr_prior_layer = Lambda(make_tcr_gaussian_prior_layer)(proc_tcr)
    tcr_prior_model = Model(inputs=input_tcr, outputs=tcr_prior_layer)
    tcr_prior_model.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_triplet = Lambda(triplet_layer_new)([proc_tcr, y_class])
    triplet = Model(inputs=[input_tcr, y_class], outputs=loss_layer_triplet)
    triplet_out = triplet([input_tcr, y_class])
    triplet.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_full = Lambda(full_layer)([triplet_out, nll_out])
    full = Model(inputs=[input_tcr, input_ep, y_class], outputs=loss_layer_full)
    full.compile(loss=identity_loss, optimizer=opt)
    full.summary()
    
    experimental_loss_layer = Lambda(experimental_layer)([proc_tcr, proc_ep[0], proc_ep[1], y_class])
    exp = Model(inputs=[input_tcr, input_ep, y_class], outputs=experimental_loss_layer)
    exp.compile(loss=identity_loss, optimizer=opt)
    model_dict = {'Encoder': nll_model_v2,
                  'Triple': triplet,
                  'NLL': nll_model,
                  'NLL_v2': nll_model_v2,
                  'TCR_Embed': tcr_network,
                  'Ep_Embed': ep_network,
                  'TCR_Gaussian_Prior': tcr_prior_model,
                  'Dist': dist_model,
                  'Full': full}
    return model_dict

def make_full_network_new_v2_archive():
    
    input_tcr = Input(shape=TCR_SHAPE)
    tcr_network = embed_TCR_network(TCR_SHAPE)
    proc_tcr = tcr_network(input_tcr)
    
    input_ep = Input(shape=EP_SHAPE)
    ep_network = make_ep_network(EP_SHAPE)
    proc_ep = ep_network(input_ep)
    
    inp_ep1 = Input(shape=EP_SHAPE)
    proc_ep1 = ep_network(inp_ep1)
    
    inp_ep2 = Input(shape=EP_SHAPE)
    proc_ep2 = ep_network(inp_ep2)
    
    inp_ep3 = Input(shape=EP_SHAPE)
    proc_ep3 = ep_network(inp_ep3)
    
    nll_loss_layer_v2 = Lambda(gaussian_nll_layer_new_v2)([proc_tcr, proc_ep1[0], proc_ep1[1], 
                                                        proc_ep2[0], proc_ep2[1], 
                                                        proc_ep3[0], proc_ep3[1]])
    nll_model_v2 = Model(inputs=[input_tcr, inp_ep1, inp_ep2, inp_ep3], outputs=nll_loss_layer_v2)
    nll_model_v2.compile(loss='categorical_crossentropy', optimizer=opt)
    #nll_out = nll_model([input_tcr, input_ep])
    #nll_out_v2 = nll_model_v2([input_tcr, inp_ep1, inp_ep2, inp_ep3])
    #input_nll = Input(shape=(3,))
    #class_out_net = make_classifier(input_nll)
    #proc_out = class_out_net(nll_out_v2)
    
    y_class = Input(shape=(1,))
    
    dist_layer = Lambda(get_dist_layer_sized)([proc_tcr, y_class])
    dist_model = Model(inputs=[input_tcr, y_class], outputs=dist_layer)
    dist_model.compile(loss=identity_loss, optimizer=opt)
    
    nll_loss_layer = Lambda(gaussian_nll_layer_new)([proc_tcr, proc_ep[0], proc_ep[1]])
    nll_model = Model(inputs=[input_tcr, input_ep], outputs=nll_loss_layer)
    nll_out = nll_model([input_tcr, input_ep])
    nll_model.compile(loss=identity_loss, optimizer=opt)
    
    tcr_prior_layer = Lambda(make_tcr_gaussian_prior_layer)(proc_tcr)
    tcr_prior_model = Model(inputs=input_tcr, outputs=tcr_prior_layer)
    tcr_prior_model.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_triplet = Lambda(triplet_layer_new)([proc_tcr, y_class])
    triplet = Model(inputs=[input_tcr, y_class], outputs=loss_layer_triplet)
    triplet_out = triplet([input_tcr, y_class])
    triplet.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_full = Lambda(full_layer)([triplet_out, nll_out])
    full = Model(inputs=[input_tcr, input_ep, y_class], outputs=loss_layer_full)
    full.compile(loss=identity_loss, optimizer=opt)
    full.summary()
    
    experimental_loss_layer = Lambda(experimental_layer)([proc_tcr, proc_ep[0], proc_ep[1], y_class])
    exp = Model(inputs=[input_tcr, input_ep, y_class], outputs=experimental_loss_layer)
    exp.compile(loss=identity_loss, optimizer=opt)
    model_dict = {'Encoder': nll_model_v2,
                  'Triple': triplet,
                  'NLL': nll_model,
                  'NLL_v2': nll_model_v2,
                  'TCR_Embed': tcr_network,
                  'Ep_Embed': ep_network,
                  'TCR_Gaussian_Prior': tcr_prior_model,
                  'Dist': dist_model,
                  'Full': full}
    return model_dict

def make_full_network():
    
    input_tcr = Input(shape=TCR_SHAPE)
    tcr_network = embed_TCR_network(TCR_SHAPE)
    proc_tcr = tcr_network(input_tcr)
    
    input_epitope = Input(shape=EP_SHAPE)
    ep_network = make_ep_network(EP_SHAPE)
    proc_ep = ep_network(input_epitope)
    
    y_class = Input(shape=(1,))
    
    dist_layer = Lambda(get_dist_layer_sized)([proc_tcr, y_class])
    dist_model = Model(inputs=[input_tcr, y_class], outputs=dist_layer)
    dist_model.compile(loss=identity_loss, optimizer=opt)
    
    nll_loss_layer = Lambda(gaussian_nll_layer)([proc_tcr, proc_ep[0], proc_ep[1]])
    model = Model(inputs=[input_tcr, input_epitope], outputs=nll_loss_layer)
    model.compile(loss=identity_loss, optimizer=opt)
    
    tcr_prior_layer = Lambda(make_tcr_gaussian_prior_layer)(proc_tcr)
    tcr_prior_model = Model(inputs=input_tcr, outputs=tcr_prior_layer)
    tcr_prior_model.compile(loss=identity_loss, optimizer=opt)
    
    loss_layer_triplet = Lambda(triplet_layer)([proc_tcr, y_class])
    triplet = Model(inputs=[input_tcr, y_class], outputs=loss_layer_triplet)
    triplet.compile(loss=identity_loss, optimizer=opt)
    
    experimental_loss_layer = Lambda(experimental_layer)([proc_tcr, proc_ep[0], proc_ep[1], y_class])
    exp = Model(inputs=[input_tcr, input_epitope, y_class], outputs=experimental_loss_layer)
    exp.compile(loss=identity_loss, optimizer=opt)
    model_dict = {'Encoder': exp,
                  'Triple': triplet,
                  'NLL': model,
                  'TCR_Embed': tcr_network,
                  'Ep_Embed': ep_network,
                  'TCR_Gaussian_Prior': tcr_prior_model,
                  'Dist': dist_model}
    return model_dict

def train(model_dict, data_dict, batch_size, epochs, period, choice_idx, name=None, auto_stop_n=None):
    exp = model_dict['Encoder']
    nll = model_dict['NLL']
    triplet = model_dict['Triple']
    dist_net = model_dict['Dist']
    n_train = data_dict['X_train'].shape[0]
    n_batches = n_train//batch_size
    n_test = data_dict['X_test'].shape[0]
    idx = np.arange(n_train)
    idx_test = np.arange(n_test)
    n_batches_test = n_test//batch_size
    X_test = data_dict['X_test']
    y_enc_test = data_dict['y_enc_test']
    y_cls_test = data_dict['y_cls_test']
    if choice_idx is None:
        X_train = data_dict['X_train']
        y_enc_train = data_dict['y_enc_train']
        y_cls_train = data_dict['y_cls_train']
    else:
        X_train = data_dict['X_train'][choice_idx]
        y_enc_train = data_dict['y_enc_train'][choice_idx]
        y_cls_train = data_dict['y_cls_train'][choice_idx]
    loss=0
    
    loss_track_val = []
    loss_track_train = []
    triplet_track_val = []
    triplet_track_train = []
    nll_track_val = []
    nll_track_train = []
    
    x_list = []
    for i in range(epochs):
        if i != 0:
            loss = np.mean(loss_list)
        print(f'Epoch {i+1}/{epochs} | Loss: {loss}', end='\r')
        np.random.shuffle(idx)
        loss_list = []
        for j in range(n_batches):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_idx = idx[start:end]
            X_batch = X_train[batch_idx]
            y_enc_batch = y_enc_train[batch_idx]
            y_cls_batch = y_cls_train[batch_idx]
            loss = exp.train_on_batch(x=[X_batch, y_enc_batch, y_cls_batch], y=y_cls_batch)
            loss_list.append(loss)
            #print(f'Epoch {i+1}/{epochs} | Batch: {j+1}/{n_batches} | Loss: {loss}', end='\r')
        if i != 0 and i % period == 0:
            x_list.append(i)
            np.random.shuffle(idx)
            loss_list_train = []
            triplet_list_train = []
            nll_list_train = []
            for j in range(n_batches):
                start = j * batch_size
                end = (j+1) * batch_size
                batch_idx = idx[start:end]
                X_batch = X_train[batch_idx]
                y_enc_batch = y_enc_train[batch_idx]
                y_cls_batch = y_cls_train[batch_idx]
                loss = exp.test_on_batch(x=[X_batch, y_enc_batch, y_cls_batch], y=y_cls_batch)
                triplet_loss = triplet.test_on_batch(x=[X_batch, y_cls_batch], y=y_cls_batch)
                #nll_loss = nll.test_on_batch(x=[X_batch, y_enc_batch], y=y_cls_batch)
                triplet_list_train.append(triplet_loss)
                #nll_list_train.append(nll_loss)
                loss_list_train.append(loss)
            
            loss_track_train.append(np.mean(loss_list_train))
            triplet_track_train.append(np.mean(triplet_list_train))
            #nll_track_train.append(np.mean(nll_list_train))
            
            np.random.shuffle(idx_test)
            loss_list_val = []
            triplet_list_val = []
            nll_list_val = []
            for k in range(n_batches_test):
                start_test = k * batch_size
                end_test = (k+1) * batch_size
                batch_idx_test = idx_test[start_test:end_test]
                X_batch_test = X_test[batch_idx_test]
                y_enc_batch_test = y_enc_test[batch_idx_test]
                y_cls_batch_test = y_cls_test[batch_idx_test]
                val = exp.test_on_batch(x=[X_batch_test, y_enc_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                triplet_loss_val = triplet.test_on_batch(x=[X_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                #nll_loss_val = nll.test_on_batch(x=[X_batch_test, y_enc_batch_test], y=y_cls_batch_test)
                triplet_list_val.append(triplet_loss_val)
                #nll_list_val.append(nll_loss_val)
                loss_list_val.append(val)
            
            loss_track_val.append(np.mean(loss_list_val))
            triplet_track_val.append(np.mean(triplet_list_val))
            #nll_track_val.append(np.mean(nll_list_val))
            
            plt.figure(figsize=(20,10))
            plt.subplot(121)
            plt.plot(x_list, loss_track_train, label='Train Total Loss')
            plt.plot(x_list, triplet_track_train, label='Train Triplet Loss')
            #plt.plot(x_list, nll_track_train, label='Train NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            plt.subplot(122)
            plt.plot(x_list, loss_track_val, label='Val Total Loss')
            plt.plot(x_list, triplet_track_val, label='Val Triplet Loss')
            #plt.plot(x_list, nll_track_val, label='Val NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            if name:
                plt.savefig(name)
            else:
                plt.savefig('test.png')
            plt.close()
            if auto_stop_n:
                if len(loss_track_val) > 2*auto_stop_n:
                    immediate = np.mean(loss_track_val[-auto_stop_n:])
                    past = np.mean(loss_track_val[-2*auto_stop_n:-auto_stop_n])
                    if immediate >= past:
                        break
                        
def train_notest(model_dict, data_dict, batch_size, epochs, period, choice_idx, name=None, auto_stop_n=None):
    exp = model_dict['Encoder']
    nll = model_dict['NLL']
    triplet = model_dict['Triple']
    dist_net = model_dict['Dist']
    n_train = data_dict['X_train'].shape[0]
    n_batches = n_train//batch_size
    #n_test = data_dict['X_test'].shape[0]
    idx = np.arange(n_train)
    #idx_test = np.arange(n_test)
    #n_batches_test = n_test//batch_size
    #X_test = data_dict['X_test']
    #y_enc_test = data_dict['y_enc_test']
    #y_cls_test = data_dict['y_cls_test']
    if choice_idx is None:
        X_train = data_dict['X_train']
        y_enc_train = data_dict['y_enc_train']
        y_cls_train = data_dict['y_cls_train']
    else:
        X_train = data_dict['X_train'][choice_idx]
        y_enc_train = data_dict['y_enc_train'][choice_idx]
        y_cls_train = data_dict['y_cls_train'][choice_idx]
    loss=0
    
    loss_track_val = []
    loss_track_train = []
    triplet_track_val = []
    triplet_track_train = []
    nll_track_val = []
    nll_track_train = []
    
    x_list = []
    for i in range(epochs):
        if i != 0:
            loss = np.mean(loss_list)
        print(f'Epoch {i+1}/{epochs} | Loss: {loss}', end='\r')
        np.random.shuffle(idx)
        loss_list = []
        for j in range(n_batches):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_idx = idx[start:end]
            X_batch = X_train[batch_idx]
            y_enc_batch = y_enc_train[batch_idx]
            y_cls_batch = y_cls_train[batch_idx]
            loss = exp.train_on_batch(x=[X_batch, y_enc_batch, y_cls_batch], y=y_cls_batch)
            loss_list.append(loss)
            #print(f'Epoch {i+1}/{epochs} | Batch: {j+1}/{n_batches} | Loss: {loss}', end='\r')
        if i != 0 and i % period == 0 and i < 0:
            x_list.append(i)
            np.random.shuffle(idx)
            loss_list_train = []
            triplet_list_train = []
            nll_list_train = []
            for j in range(n_batches):
                start = j * batch_size
                end = (j+1) * batch_size
                batch_idx = idx[start:end]
                X_batch = X_train[batch_idx]
                y_enc_batch = y_enc_train[batch_idx]
                y_cls_batch = y_cls_train[batch_idx]
                loss = exp.test_on_batch(x=[X_batch, y_enc_batch, y_cls_batch], y=y_cls_batch)
                triplet_loss = triplet.test_on_batch(x=[X_batch, y_cls_batch], y=y_cls_batch)
                #nll_loss = nll.test_on_batch(x=[X_batch, y_enc_batch], y=y_cls_batch)
                triplet_list_train.append(triplet_loss)
                #nll_list_train.append(nll_loss)
                loss_list_train.append(loss)
            
            loss_track_train.append(np.mean(loss_list_train))
            triplet_track_train.append(np.mean(triplet_list_train))
            #nll_track_train.append(np.mean(nll_list_train))
            
            np.random.shuffle(idx_test)
            loss_list_val = []
            triplet_list_val = []
            nll_list_val = []
            for k in range(n_batches_test):
                start_test = k * batch_size
                end_test = (k+1) * batch_size
                batch_idx_test = idx_test[start_test:end_test]
                X_batch_test = X_test[batch_idx_test]
                y_enc_batch_test = y_enc_test[batch_idx_test]
                y_cls_batch_test = y_cls_test[batch_idx_test]
                val = exp.test_on_batch(x=[X_batch_test, y_enc_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                triplet_loss_val = triplet.test_on_batch(x=[X_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                #nll_loss_val = nll.test_on_batch(x=[X_batch_test, y_enc_batch_test], y=y_cls_batch_test)
                triplet_list_val.append(triplet_loss_val)
                #nll_list_val.append(nll_loss_val)
                loss_list_val.append(val)
            
            loss_track_val.append(np.mean(loss_list_val))
            triplet_track_val.append(np.mean(triplet_list_val))
            #nll_track_val.append(np.mean(nll_list_val))
            
            plt.figure(figsize=(20,10))
            plt.subplot(121)
            plt.plot(x_list, loss_track_train, label='Train Total Loss')
            plt.plot(x_list, triplet_track_train, label='Train Triplet Loss')
            #plt.plot(x_list, nll_track_train, label='Train NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            plt.subplot(122)
            plt.plot(x_list, loss_track_val, label='Val Total Loss')
            plt.plot(x_list, triplet_track_val, label='Val Triplet Loss')
            #plt.plot(x_list, nll_track_val, label='Val NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            if name:
                plt.savefig(name)
            else:
                plt.savefig('test.png')
            plt.close()
            if auto_stop_n:
                if len(loss_track_val) > 2*auto_stop_n:
                    immediate = np.mean(loss_track_val[-auto_stop_n:])
                    past = np.mean(loss_track_val[-2*auto_stop_n:-auto_stop_n])
                    if immediate >= past:
                        break

def train_v2(model_dict, data_dict, batch_size, epochs, period, ep_dict):
    exp = model_dict['Encoder']
    nll = model_dict['NLL']
    triplet = model_dict['Triple']
    dist_net = model_dict['Dist']
    n_train = data_dict['X_train'].shape[0]
    n_batches = n_train//batch_size
    n_test = data_dict['X_test'].shape[0]
    idx = np.arange(n_train)
    idx_test = np.arange(n_test)
    n_batches_test = n_test//batch_size
    X_test = data_dict['X_test']
    y_enc_test = data_dict['y_enc_test']
    y_cls_test = data_dict['y_hot_test']
    X_train = data_dict['X_train']
    y_enc_train = data_dict['y_enc_train']
    y_cls_train = data_dict['y_hot_train']
    loss=0
    
    loss_track_val = []
    loss_track_train = []
    triplet_track_val = []
    triplet_track_train = []
    nll_track_val = []
    nll_track_train = []
    
    ep0 = np.repeat(ep_dict[0], batch_size, axis=0)
    ep1 = np.repeat(ep_dict[1], batch_size, axis=0)
    ep2 = np.repeat(ep_dict[2], batch_size, axis=0)
    ep3 = np.repeat(ep_dict[3], batch_size, axis=0)
    ep4 = np.repeat(ep_dict[4], batch_size, axis=0)
    ep5 = np.repeat(ep_dict[5], batch_size, axis=0)
    ep6 = np.repeat(ep_dict[6], batch_size, axis=0)
    
    x_list = []
    for i in range(epochs):
        if i != 0:
            loss = np.mean(loss_list)
        print(f'Epoch {i+1}/{epochs} | Loss: {loss}', end='\r')
        np.random.shuffle(idx)
        loss_list = []
        for j in range(n_batches):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_idx = idx[start:end]
            X_batch = X_train[batch_idx]
            y_enc_batch = y_enc_train[batch_idx]
            y_cls_batch = y_cls_train[batch_idx]
            loss = exp.train_on_batch(x=[X_batch, ep0, ep1, ep2, ep3, ep4, ep5, ep6], y=y_cls_batch)
            loss_list.append(loss)
            #print(f'Epoch {i+1}/{epochs} | Batch: {j+1}/{n_batches} | Loss: {loss}', end='\r')
        if i != 0 and i % period == 0:
            x_list.append(i)
            np.random.shuffle(idx)
            loss_list_train = []
            triplet_list_train = []
            nll_list_train = []
            for j in range(n_batches):
                start = j * batch_size
                end = (j+1) * batch_size
                batch_idx = idx[start:end]
                X_batch = X_train[batch_idx]
                y_enc_batch = y_enc_train[batch_idx]
                y_cls_batch = y_cls_train[batch_idx]
                loss = exp.test_on_batch(x=[X_batch, ep0, ep1, ep2, ep3, ep4, ep5, ep6], y=y_cls_batch)
                #triplet_loss = triplet.test_on_batch(x=[X_batch, y_cls_batch], y=y_cls_batch)
                #nll_loss = nll.test_on_batch(x=[X_batch, y_enc_batch], y=y_cls_batch)
                #triplet_list_train.append(triplet_loss)
                #nll_list_train.append(nll_loss)
                loss_list_train.append(loss)
            
            loss_track_train.append(np.mean(loss_list_train))
            #triplet_track_train.append(np.mean(triplet_list_train))
            #nll_track_train.append(np.mean(nll_list_train))
            
            np.random.shuffle(idx_test)
            loss_list_val = []
            #triplet_list_val = []
            #nll_list_val = []
            for k in range(n_batches_test):
                start_test = k * batch_size
                end_test = (k+1) * batch_size
                batch_idx_test = idx_test[start_test:end_test]
                X_batch_test = X_test[batch_idx_test]
                y_enc_batch_test = y_enc_test[batch_idx_test]
                y_cls_batch_test = y_cls_test[batch_idx_test]
                val = exp.test_on_batch(x=[X_batch_test,  ep0, ep1, ep2, ep3, ep4, ep5, ep6], y=y_cls_batch_test)
                #triplet_loss_val = triplet.test_on_batch(x=[X_batch_test, y_cls_batch_test], y=y_cls_batch_test)
                #nll_loss_val = nll.test_on_batch(x=[X_batch_test, y_enc_batch_test], y=y_cls_batch_test)
                #triplet_list_val.append(triplet_loss_val)
                #nll_list_val.append(nll_loss_val)
                loss_list_val.append(val)
            
            loss_track_val.append(np.mean(loss_list_val))
            #triplet_track_val.append(np.mean(triplet_list_val))
            #nll_track_val.append(np.mean(nll_list_val))
            
            plt.figure(figsize=(20,10))
            plt.subplot(121)
            plt.plot(x_list, loss_track_train, label='Train Total Loss')
            #plt.plot(x_list, triplet_track_train, label='Train Triplet Loss')
            #plt.plot(x_list, nll_track_train, label='Train NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            plt.subplot(122)
            plt.plot(x_list, loss_track_val, label='Val Total Loss')
            #plt.plot(x_list, triplet_track_val, label='Val Triplet Loss')
            #plt.plot(x_list, nll_track_val, label='Val NLL Loss')
            plt.legend()
            plt.xlabel('Epoch', fontsize=20)
            
            plt.savefig('test.png')
            plt.close()