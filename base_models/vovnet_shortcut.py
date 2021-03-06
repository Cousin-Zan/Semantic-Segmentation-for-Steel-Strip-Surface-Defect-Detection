"""
The implementation of VovNet101/152 based on Tensorflow.
Some codes are based on official tensorflow source codes.

@Author: Zan Peng
@Github: https://github.com/Cousin-Zan

"""

from utils.layers import Concatenate
import tensorflow as tf

layers = tf.keras.layers
backend = tf.keras.backend


class VovNet_shortcut(object):
    def __init__(self,
                 version='VovNet57_shortcut',
                 **kwargs):

        super(VovNet_shortcut, self).__init__(**kwargs)
        config_stage_ch = {'VovNet57_shortcut': [16, 16, 16, 16]}
        config_concat_ch = {'VovNet57_shortcut': [112, 208, 352, 544]}
        block_per_stage = {'VovNet57_shortcut': 1}
        layer_per_block = {'VovNet57_shortcut': [4, 6, 9, 12]}
        self.version = version
        assert version in config_stage_ch
        assert version in config_concat_ch
        assert version in block_per_stage
        assert version in layer_per_block
        self.config_stage_ch = config_stage_ch[version]
        self.config_concat_ch = config_concat_ch[version]
        self.block_per_stage = block_per_stage[version]
        self.layer_per_block = layer_per_block[version]

    def _conv3x3(self, x, out_channels, model_name, postfix,
                 stride=1, kernel_size=3, padding='same'):
        """3x3 convolution with padding"""

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        # Conv-BN-ReLU
        x1 = layers.Conv2D(out_channels, kernel_size=kernel_size,
                           strides=stride,
                           padding=padding,
                           use_bias=False,
                           name=model_name + postfix + '_conv0')(x)
        x1 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=model_name + postfix + '_bn0')(x1)
        x1 = layers.Activation(
            'relu', name=model_name + postfix + '_relu0')(x1)

        return x1

    def _conv1x1(self, x, out_channels, model_name, postfix,
                 stride=1, kernel_size=1, padding='valid'):

        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        # Conv-BN-ReLU
        x1 = layers.Conv2D(out_channels, kernel_size=kernel_size,
                           strides=stride,
                           padding=padding,
                           use_bias=False,
                           name=model_name + postfix + '_conv1')(x)
        x1 = layers.BatchNormalization(axis=bn_axis,
                                       epsilon=1.001e-5,
                                       name=model_name + postfix + '_bn1')(x1)
        x1 = layers.Activation(
            'relu', name=model_name + postfix + '_relu1')(x1)

        return x1

    def _OSA_model(self, x,
                   stage_ch,
                   concat_ch,
                   layer_per_block,
                   module_name,
                   identity=False):

        identity_feat = x
        output = []
        output.append(x)
        for i in range(layer_per_block):
            x = self._conv3x3(x, stage_ch, module_name, str(i))
            output.append(x)

        x = tf.keras.layers.concatenate(output)

        if identity:
            identity_feat = self._conv1x1(identity_feat, concat_ch, module_name, '_shortcut_')
            x = layers.add([x, identity_feat])

        return x

    def _OSA_stage(self,
                   x,
                   stage_ch,
                   concat_ch,
                   block_per_stage,
                   layer_per_block,
                   stage_num):

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        for i in range(block_per_stage):
            module_name = 'OSA' + str(stage_num) + '_' + str(i+1)
            x = self._OSA_model(x,
                                stage_ch,
                                concat_ch,
                                layer_per_block,
                                module_name,
                                identity=True)
        return x

    def __call__(self, inputs, output_stages='c5', **kwargs):
        """
        call for DenseNet.
        :param inputs: a 4-D tensor.
        :param output_stages: str or a list of str containing the output stages.
        :param kwargs: other parameters.
        :return: the output of different stages.
        """

        # stem
        x = self._conv3x3(inputs, 48, 'stem', '1', 1)
        x = self._conv3x3(x, 48, 'stem', '2', 1)
        x = self._conv3x3(x, 48, 'stem', '3', 1)

        c1 = x

        # stage2
        x = self._OSA_stage(x,
                            self.config_stage_ch[0],
                            self.config_concat_ch[0],
                            self.block_per_stage,
                            self.layer_per_block[0],
                            2)
        c2 = x

        # stage3
        x = self._OSA_stage(x,
                            self.config_stage_ch[1],
                            self.config_concat_ch[1],
                            self.block_per_stage,
                            self.layer_per_block[1],
                            3)
        c3 = x

        # stage4
        x = self._OSA_stage(x,
                            self.config_stage_ch[2],
                            self.config_concat_ch[2],
                            self.block_per_stage,
                            self.layer_per_block[2],
                            4)
        c4 = x

        # stage5
        x = self._OSA_stage(x,
                            self.config_stage_ch[3],
                            self.config_concat_ch[3],
                            self.block_per_stage,
                            self.layer_per_block[3],
                            5)
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
