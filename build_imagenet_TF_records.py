from datetime import datetime
import os
import random
import sys
import threading
import argparse

import numpy as np
import tensorflow as tf

class INVALID_JPEG(Exception):
    pass

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label):
    """Build an Example proto for an example.

    Args:
      image_buffer: string, JPEG encoding of image
      label: integer, identifier for the ground truth for the network
    Returns:
      Example proto
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'img_raw': _bytes_feature(image_buffer)}))
    return example


class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data)

    def decode_jpeg(self, image_data):
        """Check if image is valid JPEG."""
        try:
            image = self._sess.run(self._decode_jpeg,
                                   feed_dict={self._decode_jpeg_data: image_data})
            return image

        except:
            raise INVALID_JPEG


def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
      channels: integer, number of channels
    """
    # Read the image file.
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    # Decode the JPEG.
    try:
        coder.decode_jpeg(image_data)
    except INVALID_JPEG:
        print("Image not JPEG. Skipping file %s." % filename)
        return None

    return image_data


def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               labels, num_shards):
    """Processes and saves list of images as TFRecord in 1 thread.

    Args:
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
      thread_index: integer, unique batch to run index is within [0, len(ranges)).
      ranges: list of pairs of integers specifying ranges of each batches to
        analyze in parallel.
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      labels: list of integer; each integer identifies the ground truth
      num_shards: integer number of shards for this data set.
    """
    # Each thread produces N shards where N = int(num_shards / num_threads).
    # For instance, if num_shards = 128, and the num_threads = 2, then the first
    # thread would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join(args.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in files_in_shard:
            filename = filenames[i]
            label = labels[i]

            image_buffer = _process_image(filename, coder)

            if image_buffer is not None:
                example = _convert_to_example(image_buffer, label)
                writer.write(example.SerializeToString())
                shard_counter += 1
                counter += 1

                if not counter % 1000:
                    print('%s [thread %d]: Processed %d of %d images in thread batch.' %
                          (datetime.now(), thread_index, counter, num_files_in_thread))
                    sys.stdout.flush()

        writer.close()
        print('%s [thread %d]: Wrote %d images to %s' %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
    print('%s [thread %d]: Wrote %d images to %d shards.' %
          (datetime.now(), thread_index, counter, num_files_in_thread))
    sys.stdout.flush()


def _process_image_files(name, filenames, labels, num_shards, args):
    """Process and save list of images as TFRecord of Example protos.

    Args:
      name: string, unique identifier specifying the data set
      filenames: list of strings; each string is a path to an image file
      labels: list of integer; each integer identifies the ground truth
        list might contain from 0+ entries corresponding to the number of bounding
        box annotations for the image.
      num_shards: integer number of shards for this data set.
    """
    assert len(filenames) == len(labels)

    # Break all images into batches with a [ranges[i][0], ranges[i][1]].
    spacing = np.linspace(0, len(filenames), args.num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # Launch a thread for each batch.
    print('Launching %d threads for spacings: %s' % (args.num_threads, ranges))
    sys.stdout.flush()

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a generic TensorFlow-based utility for converting all image codings.
    coder = ImageCoder()

    threads = []
    for thread_index in range(len(ranges)):
        fn_args = (coder, thread_index, ranges, name, filenames, labels, num_shards)
        t = threading.Thread(target=_process_image_files_batch, args=fn_args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print('%s: Finished writing all %d images in data set.' %
          (datetime.now(), len(filenames)))
    sys.stdout.flush()


def _find_image_files(data_dir, labels_file):
    """Build a list of all images files and labels in the data set.

    Args:
      data_dir: string, path to the root directory of images.

        Assumes that the ImageNet data set resides in JPEG files located in
        the following directory structure.

          data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
          data_dir/n01440764/ILSVRC2012_val_00000543.JPEG

        where 'n01440764' is the unique synset label associated with these images.

      labels_file: string, path to the labels file.

        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          n01440764
          n01443537
          n01484850
        where each line corresponds to a label expressed as a synset. We map
        each synset contained in the file to an integer (based on the alphabetical
        ordering) starting with the integer 1 corresponding to the synset
        contained in the first line.

        The reason we start the integer labels at 1 is to reserve label 0 as an
        unused background class.

    Returns:
      filenames: list of strings; each string is a path to an image file.
      synsets: list of strings; each string is a unique WordNet ID.
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    challenge_synsets = [l.strip() for l in
                         tf.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(challenge_synsets)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(challenge_synsets), data_dir))
    return filenames, labels


def _process_dataset(name, directory, num_shards, args):
    """Process a complete data set and save it as a TFRecord.

    Args:
      name: string, unique identifier specifying the data set.
      directory: string, root path to the data set.
      num_shards: integer number of shards for this data set.
        'n02119022' --> 'red fox, Vulpes vulpes'
    """
    filenames, labels = _find_image_files(directory, args.labels_file)
    _process_image_files(name, filenames, labels, num_shards, args)


def main(args):
    assert not args.train_shards % args.num_threads, (
        'Please make the num_threads commensurate with train_shards')
    assert not args.val_shards % args.num_threads, (
        'Please make the num_threads commensurate with val_shards')
    print('Saving results to %s' % args.output_dir)

    # Run it!
    _process_dataset('validation', args.val_dir, args.val_shards, args)
    _process_dataset('train', args.train_dir, args.train_shards, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        type=str,
                        default='data',
                        help='Directory housing training and validation folders.')
    parser.add_argument('--train_dir',
                        type=str,
                        default=None,
                        help='Path to training data relative to --root_dir.')
    parser.add_argument('--val_dir',
                        type=str,
                        default=None,
                        help='Path to validation data relative to --root_dir.')
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Directory to write TFRecords into.')
    parser.add_argument('--train_shards',
                        type=int,
                        default=100,
                        help="Number of TFRecords to divide train data into.")
    parser.add_argument('--val_shards',
                        type=int,
                        default=100,
                        help="Number of TFRecords to divide validation data into.")
    parser.add_argument('--num_threads',
                        type=int,
                        default=8,
                        help="Number of threads used to write TFRecord files.")
    parser.add_argument('--labels_file',
                        type=str,
                        default=None,
                        help='File containing class labels')

    args = parser.parse_args()

    if args.train_dir == None:
        args.train_dir = os.path.join(args.root_dir, "train")
    else:
        args.train_dir = os.path.join(args.root_dir, args.train_dir)

    if args.val_dir == None:
        args.val_dir = os.path.join(args.root_dir, "val")
    else:
        args.val_dir = os.path.join(args.root_dir, args.val_dir)

    if args.output_dir == None:
        args.output_dir = os.path.join(args.root_dir, "tfrecords")
    else:
        args.output_dir = os.path.join(args.root_dir, args.output_dir)

    # Run it!
    main(args)


