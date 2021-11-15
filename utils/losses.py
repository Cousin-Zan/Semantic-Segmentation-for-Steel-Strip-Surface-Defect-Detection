"""
The implementation of some losses based on Tensorflow.

@Author: Yang Lu
@Author: Zan Peng
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf
import numpy as np
backend = tf.keras.backend

def mix_loss(labels, logits):
    return categorical_crossentropy_with_logits(labels, logits) + ssim_loss(labels, logits) + miou_loss(labels, logits)

def ssim_loss(labels, logits):

    logits = backend.softmax(logits)
    total_loss = 1 - tf.image.ssim(tf.cast(labels, tf.float32), tf.cast(logits, tf.float32), max_val = 1)

    return total_loss

def categorical_crossentropy_with_logits(labels, logits):

    labels = tf.cast(labels, tf.float32)
 
    logits = tf.reshape(logits, (-1, 4)) #(h,w,c)==>(h*w,c)
    epsilon = tf.constant(value=1e-10) #define epsilon

    logits = logits + epsilon  # prevent gradient lose
    labels = tf.reshape(labels, (-1, 4)) #(h,w,c)==>(h*w,c)
    softmax = tf.nn.softmax(logits) #activate logits with softmax

    cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), axis=[1])
    #compute all classes of every pixel loss, and epsilon prevents loss nan or inf

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #average all pixels loss

    return cross_entropy_mean

def dice_and_categorical_crossentropy_with_logits(labels, logits):

    labels = tf.cast(labels, tf.float32)
 
    logits = tf.reshape(logits, (-1, 4)) #(h,w,c)==>(h*w,c)
    epsilon = tf.constant(value=1e-10) #define epsilon

    logits = logits + epsilon  # prevent gradient lose
    labels = tf.reshape(labels, (-1, 4)) #(h,w,c)==>(h*w,c)
    softmax = tf.nn.softmax(logits) #activate logits with softmax

    cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), axis=[1])
    #compute all classes of every pixel loss, and epsilon prevents loss nan or inf

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #average all pixels loss

    smooth = 1e-5
    inse = tf.reduce_sum(labels * logits, axis=[1])
    l = tf.reduce_sum(logits * logits, axis=[1])
    r = tf.reduce_sum(labels * labels, axis=[1])
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice_loss = 1. - tf.reduce_mean(dice)

    return cross_entropy_mean + dice_loss


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)
        # compute ce loss
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)
        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - y_pred, gamma) * y_true, axis=-1)
        return backend.mean(backend.sum(weights * cross_entropy, axis=[1, 2]))

    return loss


def miou_loss(y_true, y_pred):

    y_pred = backend.softmax(y_pred)

    inter = y_pred * y_true
    inter = backend.sum(inter, axis=[1, 2])

    union = y_pred + y_true - (y_pred * y_true)
    union = backend.sum(union, axis=[1, 2])

    return 1-backend.mean(inter / (union + 1e-8))



def self_balanced_focal_loss(alpha=3, gamma=2.0):
    """
    Original by Yang Lu:

    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.

    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    def loss(y_true, y_pred):
        # cross entropy loss
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

        # sample weights
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)

        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.sum(final_loss, axis=[1, 2]))

    return loss
