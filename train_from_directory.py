import os
import glob
import time

import tensorflow as tf
import numpy as np
import cv2

import testnet

def data_gen(path):
    labels = []
    imgs = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    directories.sort()
    for i, directory in enumerate(directories):
        filenames = glob.iglob(os.path.join(path, directory, "*.JPEG"))
        for filename in filenames:
            labels.append(i)
            imgs.append(filename)
    return imgs, labels

def training_preprocess(image, label):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32)
    image = (image - 128.0) / 128.0  # Zero center images

    label = tf.one_hot(label, 200)
    return image, label

def val_preprocess(image, label):
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.cast(image, tf.float32)
    image = (image - 128.0) / 128.0  # Zero center images

    label = tf.one_hot(label, 200)
    return image, label

if __name__ == "__main__":
    train_path = "data/train"
    val_path = "data/val_organized"
    save_path = "logs/checkpoints"
    mode = "train"
    batch_size = 512
    num_epochs = 200
    num_workers = 4
    lr = 0.1
    if mode == "train":
        imgs, lbls = data_gen(train_path)
    else:
        imgs, lbls = data_gen(val_path)
    imgs = np.asarray(imgs)
    lbls = np.asarray(lbls)

    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    train_dataset = train_dataset.shuffle(buffer_size=100000)

    train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(
        training_preprocess, num_parallel_batches=num_workers,
        batch_size=batch_size))

    train_dataset = train_dataset.prefetch(buffer_size=100)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((imgs, lbls))
    val_dataset = val_dataset.shuffle(buffer_size=100000)

    val_dataset = val_dataset.apply(tf.contrib.data.map_and_batch(
        val_preprocess, num_parallel_batches=num_workers,
        batch_size=batch_size))

    val_dataset = val_dataset.prefetch(buffer_size=100)

    # Set up iterator and graph feeding nodes (train and val have same size/type)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    images, labels = iterator.get_next()

    # Load in our network
    logits, keep_prob = testnet.convnet(images, (64, 64, 1), 200)

    learning_rate = tf.placeholder(dtype=tf.float32)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)

    prediction = tf.argmax(logits, axis=1)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if mode == "train":
            sess.run(init_op)
            saver.restore(sess, os.path.join(save_path, "model.ckpt"))
            for i in range(num_epochs):
                print("Starting epoch %d" % (i + 1))
                # if not i % 50 and i != 0:
                #     lr /= 10
                train_init_op.run()
                try:
                    while True:
                        # Get an image tensor and print its value.
                        _, lss, step = sess.run([train_op, loss, global_step],
                                                feed_dict={learning_rate: lr,
                                                           keep_prob: 0.5})
                        if not step % 100:
                            print("Step: %d\tLoss: %f" % (step, lss))
                        # cv2.imshow("Image", image_array[0])
                        # cv2.waitKey(1)
                        # time.sleep(1)

                except tf.errors.OutOfRangeError:
                    # We have reached the end of `image_dataset`.
                    saver.save(sess, os.path.join(save_path, "model.ckpt"))
                    print("Model saved in path: %s" % save_path)

        else:
            saver.restore(sess, os.path.join(save_path, "model.ckpt"))
            val_init_op.run()
            correct_preds = 0
            total_preds = 0
            try:
                while True:
                    pred, imgs, lbls = sess.run([prediction, images, labels],
                                                {keep_prob: 1.0})
                    total_preds += len(pred)
                    correct_preds += np.sum(pred == np.argmax(lbls, axis=-1))
            except tf.errors.OutOfRangeError:
                print("Accuracy: %f" % (correct_preds/total_preds))
