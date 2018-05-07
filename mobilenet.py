import tensorflow as tf
import numpy as np


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _conv2d(x, filter_shape, strides, training, padding="SAME"):
    W = weight_variable(filter_shape)
    b = bias_variable([filter_shape[-1]])
    x = tf.nn.conv2d(x, W, strides, padding=padding)
    tf.nn.bias_add(x, b)
    x = tf.layers.batch_normalization(x, axis=-1, training=training)
    return tf.nn.relu(x)

def _depthwise_conv(Input, filter_shape, strides, training, padding="SAME"):
    W = weight_variable(filter_shape)
    b = bias_variable([np.prod(filter_shape[2:])])
    x = tf.nn.depthwise_conv2d(Input, W, strides, padding=padding)
    x = tf.nn.bias_add(x, b)
    x = tf.layers.batch_normalization(x, axis=-1, training=training)
    return tf.nn.relu(x)

def _pointwise_conv(Input, filter_shape, training, padding="SAME"):
    x = _conv2d(Input, filter_shape, (1, 1, 1, 1), training, padding=padding)
    x = tf.layers.batch_normalization(x, axis=-1, training=training)
    return tf.nn.relu(x)

def mobilenet(Input, input_shape, num_classes, training=False):
    if not tf.executing_eagerly():
        training = tf.placeholder(dtype=tf.bool)  # Sets mode of batchnorm nodes
    # Conv_1
    x = _conv2d(Input, (3, 3, input_shape[-1], 32), (1, 2, 2, 1), training)
                                                                 #(112, 112, 32)

    # DepthwiseSeparable_1
    x = _depthwise_conv(x, (3, 3, 32, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 32, 64), training)             #(112, 112, 64)

    # DepthwiseSeparable_2
    x = _depthwise_conv(x, (3, 3, 64, 1), (1, 2, 2, 1), training)
    x = _pointwise_conv(x, (1, 1, 64, 128), training)             #(56, 56, 128)

    # DepthwiseSeparable_3
    x = _depthwise_conv(x, (3, 3, 128, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 128, 128), training)            #(56, 56, 128)

    # DepthwiseSeparable_4
    x = _depthwise_conv(x, (3, 3, 128, 1), (1, 2, 2, 1), training)
    x = _pointwise_conv(x, (1, 1, 128, 256), training)            #(28, 28, 256)

    # DepthwiseSeparable_5
    x = _depthwise_conv(x, (3, 3, 256, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 256, 256), training)            #(28, 28, 256)

    # DepthwiseSeparable_6
    x = _depthwise_conv(x, (3, 3, 256, 1), (1, 2, 2, 1), training)
    x = _pointwise_conv(x, (1, 1, 256, 512), training)            #(14, 14, 512)

    # DepthwiseSeparable_7
    x = _depthwise_conv(x, (3, 3, 512, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 512, 512), training)            #(14, 14, 512)

    # DepthwiseSeparable_8
    x = _depthwise_conv(x, (3, 3, 512, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 512, 512), training)            #(14, 14, 512)

    # DepthwiseSeparable_9
    x = _depthwise_conv(x, (3, 3, 512, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 512, 512), training)            #(14, 14, 512)

    # DepthwiseSeparable_10
    x = _depthwise_conv(x, (3, 3, 512, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 512, 512), training)            #(14, 14, 512)

    # DepthwiseSeparable_11
    x = _depthwise_conv(x, (3, 3, 512, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 512, 512), training)            #(14, 14, 512)

    # DepthwiseSeparable_12
    x = _depthwise_conv(x, (3, 3, 512, 1), (1, 2, 2, 1), training)
    x = _pointwise_conv(x, (1, 1, 512, 1024), training)            #(7, 7, 1024)

    # DepthwiseSeparable_13
    x = _depthwise_conv(x, (3, 3, 1024, 1), (1, 1, 1, 1), training)
    x = _pointwise_conv(x, (1, 1, 1024, 1024), training)           #(7, 7, 1024)

    # Pooling_1
    x = tf.nn.avg_pool(x, (1, 7, 7, 1), (1, 1, 1, 1), "VALID")
    x = tf.layers.batch_normalization(x, axis=-1, training=training)
    x = tf.nn.relu(x)                                              #(1, 1, 1024)

    # FullyConnected_1
    x = tf.squeeze(tf.layers.dense(x, num_classes))         #(1, 1, num_classes)

    # # Output
    # x = tf.squeeze(tf.nn.softmax(x))                              #(num_classes)

    return x, training


if __name__ == "__main__":
    # Debug
    tf.enable_eager_execution()
    tf.Variable = tf.contrib.eager.Variable

    x, training = mobilenet(tf.random_normal((5, 224, 224, 3)),
                            input_shape=(224, 224, 3),
                            num_classes=1000)
    assert x.shape == (5, 1000)
    print(tf.argmax(x, -1))
