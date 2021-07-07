#!/usr/bin/env python
#coding:utf-8
"""
The implementation of some utils.

@Author: Yang Lu
@Rewrite: Zan Peng
@Github: https://github.com/Cousin-Zan
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from keras_preprocessing import image as keras_image
from PIL import Image
import numpy as np
import cv2


def load_image(name):
    ''' open the image and return an array'''
    img = Image.open(name)
    return np.array(img)


def resize_image(image, label, target_size=None):
    ''' resize the image and label with target size, return resized image and label'''
    if target_size is not None:
        # if user inputs target size, image will be resize with target size
        # and label will will be resize with target size by using inter nearest 
        image = cv2.resize(image, dsize=target_size[::-1])
        label = cv2.resize(label, dsize=target_size[::-1], interpolation=cv2.INTER_NEAREST)
    return image, label


def random_crop(image, label, crop_size):
    '''random crop image and label with crop size, return cropped image and label'''
    h, w = image.shape[0:2]
    crop_h, crop_w = crop_size
    # get origin image and crop size height and weight

    if h < crop_h or w < crop_w:
        # if origin image height < crop height or origin image weight < crop weight,
        # origin image will be resize with max of origin image height and crop height,
        # the weight is same as before
        image = cv2.resize(image, (max(w, crop_w), max(h, crop_h)))
        label = cv2.resize(label, (max(w, crop_w), max(h, crop_h)), interpolation=cv2.INTER_NEAREST)
        # Attention: the label will be resize by using inter nearest

    h_beg = np.random.randint(h - crop_h)
    # get a number which is in [0, h-crop_h)
    w_beg = np.random.randint(w - crop_w)
    # get a number which is in [0, w-crop_w)

    cropped_image = image[h_beg:h_beg + crop_h, w_beg:w_beg + crop_w]
    cropped_label = label[h_beg:h_beg + crop_h, w_beg:w_beg + crop_w]
    # crop image and label

    return cropped_image, cropped_label


def random_zoom(image, label, zoom_range):
    '''random zoom image and label with zoom_range, return zoomed image and label'''
    if np.ndim(label) == 2:
        # if label has two dims, it will be expand a dim.[h,w]=>[h,w,1] 
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3
    # default label has three dims

    if np.isscalar(zoom_range):
        # zoom_range is or not a scalar
        zx, zy = np.random.uniform(1 - zoom_range, 1 + zoom_range, 2)
    elif len(zoom_range) == 2:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        raise ValueError('`zoom_range` should be a float or '
                         'a tuple or list of two floats. '
                         'Received: %s' % (zoom_range,))

    image = keras_image.apply_affine_transform(image, zx=zx, zy=zy, fill_mode='nearest')
    label = keras_image.apply_affine_transform(label, zx=zx, zy=zy, fill_mode='nearest')

    return image, label


def random_brightness(image, label, brightness_range):
    '''random brightness image label with brightness_range, return image and label'''
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if brightness_range is not None:
        if isinstance(brightness_range, (tuple, list)) and len(brightness_range) == 2:
            brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        else:
            raise ValueError('`brightness_range` should be '
                             'a tuple or list of two floats. '
                             'Received: %s' % (brightness_range,))
        image = keras_image.apply_brightness_shift(image, brightness)
    return image, label


def random_horizontal_flip(image, label, h_flip):
    '''random flip image and label on horizontal, return image and label'''
    if h_flip:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label


def random_vertical_flip(image, label, v_flip):
    '''random flip image and label on vertical, return image and label'''
    if v_flip:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
    return image, label


def random_rotation(image, label, rotation_range):
    '''random rotation image and label with rotation_range, return image and label'''
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if rotation_range > 0.:
        theta = np.random.uniform(-rotation_range, rotation_range)
        # rotate it!
        image = keras_image.apply_affine_transform(image, theta=theta, fill_mode='nearest')
        label = keras_image.apply_affine_transform(label, theta=theta, fill_mode='nearest')
    return image, label


def random_channel_shift(image, label, channel_shift_range):
    '''shift the channel with channel_shift_range, return image and label'''
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if channel_shift_range > 0:
        channel_shift_intensity = np.random.uniform(-channel_shift_range, channel_shift_range)
        image = keras_image.apply_channel_shift(image, channel_shift_intensity, channel_axis=2)
    return image, label


def one_hot(label, num_classes):
    '''tranform the label to one_hot encoded, return heatmap'''
    if np.ndim(label) == 3:
        label = np.squeeze(label, axis=-1)
    assert np.ndim(label) == 2

    heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))
    for i in range(num_classes):
        heat_map[:, :, i] = np.equal(label, i).astype('float32')
    return heat_map


def decode_one_hot(one_hot_map):
    '''decode the heatmap, return the label'''
    return np.argmax(one_hot_map, axis=-1)


########################################################################################################################
# adamw utils
def get_weight_decays(model, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_l2regs = _get_layer_l2regs(layer)
        if layer_l2regs:
            for layer_l2 in layer_l2regs:
                weight_name, weight_l2 = layer_l2
                wd_dict.update({weight_name: weight_l2})
                if weight_l2 != 0 and verbose:
                    print(("WARNING: {} l2-regularization = {} - should be "
                           "set 0 before compiling model").format(
                        weight_name, weight_l2))
    return wd_dict


def fill_dict_in_order(_dict, _list_of_vals):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = _list_of_vals[idx]
    return _dict


def _get_layer_l2regs(layer):
    if hasattr(layer, 'layer') or hasattr(layer, 'cell'):
        return _rnn_l2regs(layer)
    else:
        l2_lambda_kb = []
        for weight_name in ['kernel', 'bias']:
            _lambda = getattr(layer, weight_name + '_regularizer', None)
            if _lambda is not None:
                l2_lambda_kb.append([getattr(layer, weight_name).name,
                                     float(_lambda.l2)])
        return l2_lambda_kb


def _rnn_l2regs(layer):
    _layer = layer.layer if 'backward_layer' in layer.__dict__ else layer
    cell = _layer.cell

    l2_lambda_krb = []
    if hasattr(cell, 'kernel_regularizer') or \
            hasattr(cell, 'recurrent_regularizer') or hasattr(cell, 'bias_regularizer'):
        for weight_name in ['kernel', 'recurrent', 'bias']:
            _lambda = getattr(cell, weight_name + '_regularizer', None)
            if _lambda is not None:
                weight_name = weight_name if 'recurrent' not in weight_name \
                    else 'recurrent_kernel'
                l2_lambda_krb.append([getattr(cell, weight_name).name,
                                      float(_lambda.l2)])
    return l2_lambda_krb
