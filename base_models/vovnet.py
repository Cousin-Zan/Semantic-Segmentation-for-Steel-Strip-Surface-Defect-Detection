"""
The implementation of VovNet101/152 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Zan Peng
@Github: https://github.com/Cousin-Zan

"""
from tensorflow.python.client import session
from tensorflow.python.ops.gen_array_ops import concat
from utils.layers import Concatenate
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


class VovNet(object):
    def __init__(self, version='VovNet101', dilation=None, **kwargs):
        """
        The implementation of VovNet based on Tensorflow.
        :param version: 'VovNet101' or 'VovNet152'.
        :param dilation: Whether to use dilation strategy.
        :param kwargs: other parameters.
        """
        super(VovNet, self).__init__(**kwargs)
        params = {'VovNet101': [4, 6, 33, 5],
                  'VovNet101_ese': [4, 6, 33, 5],
                  'VovNet101_ac': [4, 6, 33, 5],
                  'VovNet101_ese_ac': [4, 6, 33, 5],
                  'VovNet152': [4, 12, 53, 5]}
        self.version = version
        assert version in params
        self.params = params[version]

        if dilation is None:
            self.dilation = [1, 1]
        else:
            self.dilation = dilation
        assert len(self.dilation) == 2

    def _osa_block(self, x, blocks, name, dilation=1):
        """A osa block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        _, h, w, _ = backend.int_shape(x)
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        osa = [x]
        for i in range(blocks):
            if self.version in ['VovNet101_ac', 'VovNet101_ese_ac']:
                x = self._acb_block(x, 48, name=name + '_block' + str(i + 1), dilation=dilation)
            else:
                x = self._conv_block(x, 48, name=name + '_block' + str(i + 1), dilation=dilation)
            osa.append(x)
        output = Concatenate(out_size=(h, w), axis=bn_axis, name=name + '_concat')(osa)

        if self.version in ['VovNet101_ese', 'VovNet101_ese_ac']:
            xe = layers.GlobalAveragePooling2D(name='ese_gap' + str(i + 1))(output)
            _, xe_shape = backend.int_shape(xe)
            xe = layers.Dense(xe_shape,name='ese_fc' + str(i + 1))(xe)
            xe = layers.Activation('sigmoid',name='ese_sigmoid' + str(i + 1))(xe)

            xe = layers.multiply([output, xe])
            _,_,_, xe_shape = backend.int_shape(xe)
            x = layers.Conv2D(xe_shape, 1, strides=1, padding='same', use_bias=False, name='shotcut' + str(i + 1))(x)
            output = layers.add([xe, x])

        return output

    def _transition_block(self, x, reduction, name, dilation=1):
        """A transition block.

        # Arguments
            x: input tensor.
            reduction: float, compression rate at transition layers.
            name: string, block label.

        # Returns
            output tensor for the block.
        """
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name=name + '_bn')(x)
        x = layers.Activation('relu', name=name + '_relu')(x)
        x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                          use_bias=False,
                          name=name + '_conv',
                          dilation_rate=dilation)(x)
        if dilation == 1:
            x = layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
        return x

    def _conv_block(self, x, growth_rate, name, dilation=1):
        """A building block for a osa block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at osa layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        x1 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_0_bn')(x)
        x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
        x1 = layers.Conv2D(4 * growth_rate, 1,
                           use_bias=False,
                           name=name + '_1_conv')(x1)
        x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_1_bn')(x1)
        x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
        x1 = layers.Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv',
                           dilation_rate=dilation)(x1)
        return x1

    def _acb_block(self, x, growth_rate, name, dilation=1):
        """A building block for a osa block.

        # Arguments
            x: input tensor.
            growth_rate: float, growth rate at osa layers.
            name: string, block label.

        # Returns
            Output tensor for the block.
        """

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        branch_1 = layers.Conv2D(growth_rate, (1,3),
                           padding='same',
                           use_bias=False,
                           name=name + '_1_conv')(x)
        branch_2 = layers.Conv2D(growth_rate, 3,
                           padding='same',
                           use_bias=False,
                           name=name + '_2_conv')(x)
        branch_3 = layers.Conv2D(growth_rate, (3,1),
                           padding='same',
                           use_bias=False,
                           name=name + '_3_conv')(x)
    
        branch_1 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_1_bn')(branch_1)
        branch_2 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_2_bn')(branch_2)
        branch_3 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=name + '_3_bn')(branch_3)

        master = layers.add([branch_1, branch_2, branch_3])
        output = layers.Activation('relu', name=name + '_0_relu')(master)
        
        return output

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for DenseNet.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """
        _, h, w, _ = backend.int_shape(inputs)

        blocks = self.params
        dilation = self.dilation
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
        x = layers.Conv2D(64, 3, strides=2, use_bias=False, name='conv1/conv')(x)
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
        x = layers.Activation('relu', name='conv1/relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1')(x)
        c1 = x

        x = self._osa_block(x, blocks[0], name='conv2')
        x = self._transition_block(x, 0.5, name='pool2')
        c2 = x

        x = self._osa_block(x, blocks[1], name='conv3')
        x = self._transition_block(x, 0.5, name='pool3', dilation=dilation[0])
        c3 = x

        x = self._osa_block(x, blocks[2], name='conv4', dilation=dilation[0])
        x = self._transition_block(x, 0.5, name='pool4', dilation=dilation[1])
        c4 = x

        x = self._osa_block(x, blocks[3], name='conv5', dilation=dilation[1])
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
        x = layers.Activation('relu', name='relu')(x)
        c5 = x

        self.outputs = {'c1': c1,
                        'c2': c2,
                        'c3': c3,
                        'c4': c4,
                        'c5': c5}

        if type(output_stages) is not list:
            return self.outputs[output_stages]
        else:
            return [self.outputs[ci] for ci in output_stages]
