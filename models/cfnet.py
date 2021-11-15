"""
The implementation of PAN (Pyramid Attention Networks) based on Tensorflow.

@Author: Yang Lu
@Author: Zan Peng
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils import layers as custom_layers
from models import Network
import tensorflow as tf

from utils.layers import Concatenate

layers = tf.keras.layers
models = tf.keras.models
backend = tf.keras.backend


class CFNET(Network):
    def __init__(self, num_classes, version='CFNET', base_model='OSA', **kwargs):
        """
        The initialization of CFNET.
        :param num_classes: the number of predicted classes.
        :param version: 'CFNET'
        :param base_model: the backbone model
        :param kwargs: other parameters
        """
        base_model = 'ResNet50' if base_model is None else base_model
        assert version == 'CFNET'

        if base_model == 'OSA':
            self.up_size = [(2, 2), (2, 2), (2, 2), (2, 2)]
        else:
            raise ValueError('The base model \'{model}\' is not '
                             'supported in CFNET.'.format(model=base_model))

        super(CFNET, self).__init__(num_classes, version, base_model, **kwargs)

    def __call__(self, inputs=None, input_size=None, **kwargs):
        assert inputs is not None or input_size is not None

        if inputs is None:
            assert isinstance(input_size, tuple)
            inputs = layers.Input(shape=input_size + (3,))
        return self._cfnet(inputs)

    def _conv_bn_relu(self, x, filters, kernel_size, strides=1):
        x = layers.Conv2D(filters, kernel_size, strides, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def _fpa(self, x, out_filters):
        _, h, w, _ = backend.int_shape(x)

        glb = custom_layers.GlobalMaxPooling2D(keep_dims=True)(x)
        glb = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(glb)

        # down
        down1 = layers.MaxPooling2D(pool_size=(2, 2))(x)
        down1 = self._conv_bn_relu(down1, out_filters, 3, 1)
        down1 = self._conv_bn_relu(down1, out_filters, 3, 1)
        down1 = self._conv_bn_relu(down1, out_filters, 3, 1)

        down2 = layers.MaxPooling2D(pool_size=(2, 2))(down1)
        down2 = self._conv_bn_relu(down2, out_filters, 3, 1)
        down2 = self._conv_bn_relu(down2, out_filters, 3, 1)

        down3 = layers.MaxPooling2D(pool_size=(2, 2))(down2)
        down3 = self._conv_bn_relu(down3, out_filters, 3, 1)

        down1 = self._conv_bn_relu(down1, out_filters, 3, 1)
        down1 = self._conv_bn_relu(down1, out_filters, 3, 1)
        down1 = self._conv_bn_relu(down1, out_filters, 3, 1)

        down2 = self._conv_bn_relu(down2, out_filters, 3, 1)
        down2 = self._conv_bn_relu(down2, out_filters, 3, 1)

        down3 = self._conv_bn_relu(down3, out_filters, 3, 1)

        # up
        up2 = layers.UpSampling2D(size=(2, 2))(down3)
        up2 = layers.Add()([up2, down2])

        up1 = layers.UpSampling2D(size=(2, 2))(up2)
        up1 = layers.Add()([up1, down1])

        up = layers.UpSampling2D(size=(2, 2))(up1)

        x = layers.Conv2D(out_filters, 1, strides=1, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        # multiply
        x = layers.Multiply()([x, up])

        # add
        x = layers.Add()([x, glb])

        return x

    def _gau(self, x, y, out_filters, up_size=(2, 2)):
        glb = custom_layers.GlobalAveragePooling2D(keep_dims=True)(y)
        glb = layers.Conv2D(out_filters, 1, strides=1, activation='sigmoid', kernel_initializer='he_normal')(glb)

        x = self._conv_bn_relu(x, out_filters, 3, 1)
        x = layers.Multiply()([x, glb])

        # y = layers.UpSampling2D(size=up_size, interpolation='bilinear')(y)
        y = layers.Conv2DTranspose(out_filters, 3, strides=2, padding='same', use_bias=False)(y)

        y = layers.Add()([x, y])

        return y

    def _fmpm(self, inputs):
        num_classes = self.num_classes
        _, h, w, _ = backend.int_shape(inputs)
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        conv01 = self._conv_bn_relu(inputs, 64, 3, 1)
        conv11 = self._conv_bn_relu(conv01, 64, 3, 1)
        mp1 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(conv11)

        conv21 = self._conv_bn_relu(mp1, 64, 3, 1)
        mp2 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(conv21)

        conv31 = self._conv_bn_relu(mp2, 64, 3, 1)
        mp3 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(conv31)

        conv41 = self._conv_bn_relu(mp3, 64, 3, 1)
        mp4 = layers.MaxPool2D(pool_size=2, strides=2, padding='same')(conv41)

        conv51 = self._conv_bn_relu(mp4, 64, 3, 1)
        up1 = layers.UpSampling2D(size=2, interpolation='bilinear')(conv51)

        c1 = Concatenate(out_size=(h, w), axis=bn_axis)([conv41, up1])
        conv61 = self._conv_bn_relu(c1, 64, 3, 1)
        up2 = layers.UpSampling2D(size=2, interpolation='bilinear')(conv61)

        c2 = Concatenate(out_size=(h, w), axis=bn_axis)([conv31, up2])
        conv71 = self._conv_bn_relu(c2, 64, 3, 1)
        up3 = layers.UpSampling2D(size=2, interpolation='bilinear')(conv71)

        c3 = Concatenate(out_size=(h, w), axis=bn_axis)([conv21, up3])
        conv81 = self._conv_bn_relu(c3, 64, 3, 1)
        up4 = layers.UpSampling2D(size=2, interpolation='bilinear')(conv81)

        c4 = Concatenate(out_size=(h, w), axis=bn_axis)([conv11, up4])
        conv91 = self._conv_bn_relu(c4, 64, 3, 1)
        conv92 = self._conv_bn_relu(conv91, num_classes, 3, 1)

        return conv92


    def _cfnet(self, inputs):
        num_classes = self.num_classes
        up_size = self.up_size
        _, h, w, _ = backend.int_shape(inputs)
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        c1, c2, c3, c4, c5 = self.encoder(inputs, output_stages=['c1', 'c2', 'c3', 'c4', 'c5'])

        y = self._fpa(c5, 544)

        y = self._gau(c4, y, 352, up_size[0])
        y = self._gau(c3, y, 208, up_size[1])
        y = self._gau(c2, y, 112, up_size[2])
        y = self._gau(c1, y, 48, up_size[3])

        y = self._conv_bn_relu(y, num_classes, 1, 1)
        # y = layers.UpSampling2D(size=up_size[3], interpolation='bilinear',name='coarse')(y)

        cmpm_output = Concatenate(out_size=(h, w), axis=bn_axis)([y]*8)
        outputs = self._fmpm(cmpm_output)

        print(y)
        print(outputs)

        return models.Model(inputs, [y, outputs], name=self.version)
