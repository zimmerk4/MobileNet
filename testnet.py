import tensorflow as tf
import numpy as np

def regression(inputs, input_shape, num_classes):
    in_flat = tf.reshape(inputs, shape=(-1, np.prod(input_shape)))
    fc = tf.layers.Dense(units=num_classes, activation=tf.nn.relu)
    x = fc(in_flat)
    return x

def convnet(inputs, input_shape, num_classes):
    keep_prob = tf.placeholder(dtype=tf.float32)
    x = tf.layers.conv2d(inputs, 32, (3, 3), (2, 2), padding="SAME", activation=tf.nn.relu)  #(32, 32, 32)
    x = tf.layers.conv2d(x, 64, (3, 3), (2, 2), padding="SAME", activation=tf.nn.relu)  #(16, 16, 64)
    x = tf.layers.conv2d(x, 128, (3, 3), (2, 2), padding="SAME", activation=tf.nn.relu)  #(8, 8, 128)
    x = tf.reshape(x, (-1, 8*8*128))
    x = tf.layers.dropout(x, keep_prob)
    x = tf.layers.dense(x, 200, activation=tf.nn.relu)
    return x, keep_prob


if __name__ == "__main__":
    tf.enable_eager_execution()
    print(convnet(tf.random_uniform((5, 64, 64, 1)), (64, 64, 1), 200).shape)