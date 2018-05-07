import os
import glob
import time

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

import mobilenet
# tf.enable_eager_execution()
IMG_SHAPE = [56, 56, 1]
NUM_CLASSES = 201  # Class 0 reserved for background


def decode_example(serialized_example):
    """ Decode a TF Example into a floating point image and onehot label."""
    features = {'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)}
    record = tf.parse_single_example(serialized_example,
                                     features=features)
    image = tf.image.decode_jpeg(record['img_raw'], channels=IMG_SHAPE[-1])
    image = tf.cast(image, tf.float32)
    label = tf.cast(record['label'], tf.int64)
    # label = tf.one_hot(label, depth=NUM_CLASSES)

    return image, label


def training_preprocess(serialized_example):
    image, label = decode_example(serialized_example)
    crop_image = tf.random_crop(image, IMG_SHAPE)
    flip_image = tf.image.random_flip_left_right(crop_image)
    centered_image = (flip_image - 128.0) / 128.0

    return centered_image, label

def val_preprocess(serialized_example):
    image, label = decode_example(serialized_example)
    crop_image = tf.image.resize_image_with_crop_or_pad(image, IMG_SHAPE[:2])
    centered_image = (crop_image - 128.0) / 128.0
    return centered_image, label

def main():
    # Basic training params
    batch_size = 4
    num_epochs = 10
    lr = 0.0001
    num_workers = 4

    # Data locations
    root_dir = "/home/kyle/PycharmProjects/MobileNet/data"
    data_dir = os.path.join(root_dir, "tfrecords")
    log_dir= os.path.join(os.getcwd(), "logs")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  # make sure this exists...

    # Load our model into a tf.Graph
    graph = tf.Graph()
    with graph.as_default():
        # Get the list of filenames and corresponding list of labels for training et validation
        train_filenames = os.path.join(data_dir, "train*")
        val_filenames = os.path.join(data_dir, "validation*")

        # Training dataset
        train_dataset = tf.data.Dataset.list_files(train_filenames)
        train_dataset = train_dataset.shuffle(buffer_size=100000)

        train_dataset = train_dataset.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_workers))

        train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
            training_preprocess, num_parallel_batches=num_workers,
            batch_size=batch_size))

        train_dataset = train_dataset.prefetch(buffer_size=100)

        # Validation dataset
        val_dataset = tf.data.Dataset.list_files(val_filenames)
        val_dataset = val_dataset.shuffle(buffer_size=100000)

        val_dataset = val_dataset.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=num_workers))

        val_dataset = val_dataset.apply(tf.contrib.data.map_and_batch(
            training_preprocess, num_parallel_batches=num_workers,
            batch_size=batch_size))

        val_dataset = val_dataset.prefetch(buffer_size=100)

        # Set up iterator and graph feeding nodes
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)
        train_init_op = iterator.make_initializer(train_dataset)
        val_init_op = iterator.make_initializer(val_dataset)

        images, labels = iterator.get_next()

        import testnet
        logits = testnet.testnet(images, IMG_SHAPE, NUM_CLASSES)
        # logits, training = mobilenet.mobilenet(images, IMG_SHAPE, NUM_CLASSES)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                             labels=labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(loss, tf.train.get_global_step())

        # Evaluation metrics
        prediction = tf.argmax(logits, 1)
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Finalize graph
        init_op = tf.global_variables_initializer()
        tf.get_default_graph().finalize()
        # --------------------------------------------------------------------------

        with tf.Session(graph=graph) as sess:
            sess.run(init_op)

            # Train the entire model for a few more epochs, continuing with the *same* weights.
            for epoch in range(num_epochs):
                print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
                sess.run(train_init_op)
                counter = 0
                while True:
                    try:
                        if counter % 1 == 0:
                        # if 0:
                            lss, acc, _ = sess.run([loss, accuracy, train_op])
                            # Check accuracy on the train and val sets every epoch
                            print("Loss: %f" % lss,'\tTrain accuracy: %f' % acc.mean())
                            counter = 0
                        else:
                            lgts, lbls, imgs = sess.run([logits, labels, images])
                            cv2.imshow("Images", imgs[0])
                            cv2.waitKey(1)
                            time.sleep(1)
                    except tf.errors.OutOfRangeError:
                        break
                    counter += 1


if __name__ == '__main__':
    main()