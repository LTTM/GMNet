# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import time
import numpy as np
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import quantize as contrib_quantize
from tensorflow.contrib import training as contrib_training
from research.deeplab import common
from research.deeplab import model
from research.deeplab import model_softmax
from research.deeplab import model_edge
from research.deeplab import model_softmax_edge
from research.deeplab import model_features
from research.deeplab.datasets import data_generator
from research.deeplab.utils import save_annotation

from datetime import datetime
from research.deeplab.utils.general_utils import *

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1, 'The number of images in each batch during evaluation.')

flags.DEFINE_list('vis_crop_size', '513,513', 'Crop size [height, width] for visualization.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5, 'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None, 'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16, 'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0], 'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False, 'Add flipped images for evaluation or not.')

flags.DEFINE_integer('quantize_delay_step', -1, 'Steps to start quantized training. If < 0, will not quantize model.')

# Dataset settings.
# 58 parts
flags.DEFINE_string('dataset', 'pascal_voc_seg_58_parts', 'Name of the segmentation dataset.')

# 21 parts
# flags.DEFINE_string('dataset', 'pascal_voc_seg', 'Name of the segmentation dataset.')

flags.DEFINE_string('vis_split', 'val', 'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes', 'ade20k'], 'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', False, 'Also save raw predictions.')

flags.DEFINE_integer('max_number_of_iterations', 0, 'Maximum number of visualization iterations. Will loop '
                                                    'indefinitely upon nonpositive values.')

flags.DEFINE_integer('save_softmax', 0, 'Also save softmax predictions.')

flags.DEFINE_integer('model_softmax', 0, 'Use model_softmax or not (Default use model)')

flags.DEFINE_integer('model_edge', 0, 'Use model_edge or not (Default use model)')

flags.DEFINE_integer('model_softmax_edge', 0, 'Use model_softmax_edge or not')

flags.DEFINE_integer('model_features', 0, 'Use model_features or not')

flags.DEFINE_integer('load_macro_classes_features', 0, 'Load macro classes features')

flags.DEFINE_integer('save_parts_features', 0, 'Also save parts features.')

# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The folder where softmax semantic segmentation predictions are saved.
_SOFTMAX_SAVE_FOLDER = 'softmax_results'

_PARTS_FEATURES_SAVE_FOLDER = 'parts_features_results'

# Suffix for the softmax results
_PROB_SUFFIX = '_prob'

# The format to save image.
# _IMAGE_FORMAT = '%06d_image'
_IMAGE_FORMAT = '%s_image'

# The format to save prediction
# _PREDICTION_FORMAT = '%06d_prediction'
_PREDICTION_FORMAT = '%s_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
    """Converts the predicted label for evaluation.

  There are cases where the training labels are not equal to the evaluation
  labels. This function is used to perform the conversion so that we could
  evaluate the results on the evaluation server.

  Args:
    prediction: Semantic segmentation prediction.
    train_id_to_eval_id: A list mapping from train id to evaluation id.

  Returns:
    Semantic segmentation prediction whose labels have been changed.
  """
    converted_prediction = prediction.copy()
    for train_id, eval_id in enumerate(train_id_to_eval_id):
        converted_prediction[prediction == train_id] = eval_id

    return converted_prediction


def _process_batch(sess,
                   original_images,
                   semantic_predictions,
                   softmax_predictions,
                   parts_features,
                   image_names,
                   image_heights,
                   image_widths,
                   image_id_offset,
                   save_dir,
                   raw_save_dir,
                   softmax_dir,
                   parts_features_dir,
                   train_id_to_eval_id=None):
    """Evaluates one single batch qualitatively.

  Args:
    sess: TensorFlow session.
    original_images: One batch of original images.
    semantic_predictions: One batch of semantic segmentation predictions.
    image_names: Image names.
    image_heights: Image heights.
    image_widths: Image widths.
    image_id_offset: Image id offset for indexing images.
    save_dir: The directory where the predictions will be saved.
    raw_save_dir: The directory where the raw predictions will be saved.
    train_id_to_eval_id: A list mapping from train id to eval id.
  """
    parts_features_run = None

    if FLAGS.model_features == 1:
        (original_images,
         semantic_predictions,
         softmax_predictions,
         parts_features_run,
         image_names,
         image_heights,
         image_widths) = sess.run([original_images, semantic_predictions, softmax_predictions, parts_features,
                                   image_names, image_heights, image_widths])
    else:
        (original_images,
         semantic_predictions,
         softmax_predictions,
         image_names,
         image_heights,
         image_widths) = sess.run([original_images, semantic_predictions, softmax_predictions,
                                   image_names, image_heights, image_widths])

    num_image = semantic_predictions.shape[0]
    for i in range(num_image):
        image_name = image_names[i]
        print_message(FilesName.VIS, image_name)
        image_name = image_name.decode("utf-8")
        image_height = np.squeeze(image_heights[i])
        image_width = np.squeeze(image_widths[i])
        original_image = np.squeeze(original_images[i])
        semantic_prediction = np.squeeze(semantic_predictions[i])
        softmax_prediction = np.squeeze(softmax_predictions[i])
        crop_semantic_prediction = semantic_prediction[:image_height, :image_width]
        crop_softmax_prediction = softmax_prediction[:image_height, :image_width]

        if parts_features_run is not None:
            parts_features_run = np.squeeze(parts_features_run[i])

        # Save image.
        if FLAGS.vis_batch_size == 1:
            image_filename = _IMAGE_FORMAT % image_name
        else:
            image_filename = _IMAGE_FORMAT % (image_id_offset + i)

        save_annotation.save_annotation(original_image, save_dir, image_filename, add_colormap=False)

        # Save prediction.
        if FLAGS.vis_batch_size == 1:
            prediction_filename = _PREDICTION_FORMAT % image_name
        else:
            prediction_filename = _PREDICTION_FORMAT % (image_id_offset + i)

        save_annotation.save_annotation(crop_semantic_prediction,
                                        save_dir,
                                        prediction_filename,
                                        add_colormap=True,
                                        colormap_type=FLAGS.colormap_type)

        # Save softmax.
        if FLAGS.save_softmax == 1:
            np.save('%s/%s.npy' % (softmax_dir, image_name), crop_softmax_prediction)

        # Save features.
        if FLAGS.save_parts_features == 1:
            np.save('%s/%s.npy' % (parts_features_dir, image_name), parts_features_run)

        # Save raw predictions.
        if FLAGS.also_save_raw_predictions:
            image_filename = os.path.basename(image_name)

            if train_id_to_eval_id is not None:
                crop_semantic_prediction = _convert_train_id_to_eval_id(crop_semantic_prediction, train_id_to_eval_id)

            save_annotation.save_annotation(crop_semantic_prediction,
                                            raw_save_dir,
                                            image_filename,
                                            add_colormap=False)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print_message(FilesName.VIS, "Start at " + date_now, MessageType.HEADER)

    # Get dataset-dependent information.
    dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.vis_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=FLAGS.vis_batch_size,
        crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        model_variant=FLAGS.model_variant,
        is_training=False,
        should_shuffle=False,
        should_repeat=False)

    train_id_to_eval_id = None
    if dataset.dataset_name == data_generator.get_cityscapes_dataset_name():
        tf.logging.info('Cityscapes requires converting train_id to eval_id.')
        train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

    # Prepare for visualization.
    tf.gfile.MakeDirs(FLAGS.vis_logdir)
    save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
    tf.gfile.MakeDirs(save_dir)
    raw_save_dir = os.path.join(FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
    tf.gfile.MakeDirs(raw_save_dir)

    softmax_dir = os.path.join(FLAGS.vis_logdir, _SOFTMAX_SAVE_FOLDER)
    tf.gfile.MakeDirs(softmax_dir)

    parts_features_dir = os.path.join(FLAGS.vis_logdir, _PARTS_FEATURES_SAVE_FOLDER)
    tf.gfile.MakeDirs(parts_features_dir)

    tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

    edges = None
    parts_features = None

    with tf.Graph().as_default():
        samples = dataset.get_one_shot_iterator(use_softmax=0, use_gt_macro_classes=0, use_edges=0).get_next()

        if FLAGS.load_macro_classes_features == 0:
            samples[common.MACRO_CLASSES_FEATURES] = None

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
            crop_size=[int(sz) for sz in FLAGS.vis_crop_size],
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')

            if FLAGS.model_softmax == 1:
                predictions = model_softmax.predict_labels(samples[common.IMAGE],
                                                           samples[common.SOFTMAX],
                                                           model_options=model_options,
                                                           image_pyramid=FLAGS.image_pyramid)
            elif FLAGS.model_edge == 1:
                predictions, edges = model_edge.predict_labels(samples[common.IMAGE],
                                                               model_options=model_options,
                                                               image_pyramid=FLAGS.image_pyramid)

            elif FLAGS.model_softmax_edge == 1:
                predictions, edges = model_softmax_edge.predict_labels(samples[common.IMAGE],
                                                                       samples[common.SOFTMAX],
                                                                       model_options,
                                                                       image_pyramid=FLAGS.image_pyramid)

            elif FLAGS.model_features == 1:
                predictions, edges, parts_features = model_features.predict_labels(samples[common.IMAGE],
                                                                                   samples[
                                                                                       common.MACRO_CLASSES_FEATURES],
                                                                                   model_options,
                                                                                   image_pyramid=FLAGS.image_pyramid)

            else:
                predictions = model.predict_labels(samples[common.IMAGE],
                                                   model_options=model_options,
                                                   image_pyramid=FLAGS.image_pyramid)

        else:
            tf.logging.info('Performing multi-scale test.')
            if FLAGS.quantize_delay_step >= 0:
                raise ValueError('Quantize mode is not supported with multi-scale test.')

            if FLAGS.model_softmax == 1:
                predictions = model_softmax.predict_labels_multi_scale(samples[common.IMAGE],
                                                                       samples[common.SOFTMAX],
                                                                       model_options=model_options,
                                                                       eval_scales=FLAGS.eval_scales,
                                                                       add_flipped_images=FLAGS.add_flipped_images)
            elif FLAGS.model_edge == 1:
                predictions, edges = model_edge.predict_labels_multi_scale(samples[common.IMAGE],
                                                                           model_options=model_options,
                                                                           eval_scales=FLAGS.eval_scales,
                                                                           add_flipped_images=FLAGS.add_flipped_images)
            elif FLAGS.model_softmax_edge == 1:
                predictions, edges = model_softmax_edge.predict_labels_multi_scale(samples[common.IMAGE],
                                                                                   samples[common.SOFTMAX],
                                                                                   model_options=model_options,
                                                                                   eval_scales=FLAGS.eval_scales,
                                                                                   add_flipped_images=FLAGS.add_flipped_images)

            elif FLAGS.model_features == 1:
                predictions, edges, parts_features = model_features.predict_labels_multi_scale(
                    samples[common.IMAGE],
                    samples[common.MACRO_CLASSES_FEATURES],
                    model_options=model_options,
                    eval_scales=FLAGS.eval_scales,
                    add_flipped_images=FLAGS.add_flipped_images)

            else:
                predictions = model.predict_labels_multi_scale(samples[common.IMAGE],
                                                               model_options=model_options,
                                                               eval_scales=FLAGS.eval_scales,
                                                               add_flipped_images=FLAGS.add_flipped_images)

        predictions_argmax = predictions[common.OUTPUT_TYPE]
        predictions_softmax = predictions[common.OUTPUT_TYPE + _PROB_SUFFIX]

        if FLAGS.min_resize_value and FLAGS.max_resize_value:
            # Only support batch_size = 1, since we assume the dimensions of original
            # image after tf.squeeze is [height, width, 3].
            assert FLAGS.vis_batch_size == 1

            # Reverse the resizing and padding operations performed in preprocessing.
            # First, we slice the valid regions (i.e., remove padded region) and then
            # we resize the predictions back.
            original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
            original_image_shape = tf.shape(original_image)
            predictions_argmax = tf.slice(predictions_argmax,
                                          [0, 0, 0],
                                          [1, original_image_shape[0],
                                           original_image_shape[1]])

            resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]), tf.squeeze(samples[common.WIDTH])])

            predictions_argmax = tf.squeeze(tf.image.resize_images(tf.expand_dims(predictions_argmax, 3),
                                                                   resized_shape,
                                                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                                                   align_corners=True), 3)


        tf.train.get_or_create_global_step()
        if FLAGS.quantize_delay_step >= 0:
            contrib_quantize.create_eval_graph()

        num_iteration = 0
        max_num_iteration = FLAGS.max_number_of_iterations

        checkpoints_iterator = contrib_training.checkpoints_iterator(FLAGS.checkpoint_dir,
                                                                     min_interval_secs=FLAGS.eval_interval_secs)

        for checkpoint_path in checkpoints_iterator:
            num_iteration += 1
            tf.logging.info('Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
            tf.logging.info('Visualizing with model %s', checkpoint_path)

            scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=scaffold,
                master=FLAGS.master,
                checkpoint_filename_with_path=checkpoint_path)

            with tf.train.MonitoredSession(session_creator=session_creator, hooks=None) as sess:
                batch = 0
                image_id_offset = 0

                while not sess.should_stop():
                    print_message(FilesName.VIS, "Iteration " + str(batch))

                    tf.logging.info('Visualizing batch %d', batch + 1)
                    _process_batch(sess=sess,
                                   original_images=samples[common.ORIGINAL_IMAGE],
                                   semantic_predictions=predictions_argmax,
                                   softmax_predictions=predictions_softmax,
                                   parts_features=parts_features,
                                   image_names=samples[common.IMAGE_NAME],
                                   image_heights=samples[common.HEIGHT],
                                   image_widths=samples[common.WIDTH],
                                   image_id_offset=image_id_offset,
                                   save_dir=save_dir,
                                   raw_save_dir=raw_save_dir,
                                   softmax_dir=softmax_dir,
                                   parts_features_dir=parts_features_dir,
                                   train_id_to_eval_id=train_id_to_eval_id)

                    image_id_offset += FLAGS.vis_batch_size
                    batch += 1

            tf.logging.info('Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

            if max_num_iteration > 0 and num_iteration >= max_num_iteration:
                break

    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print_message(FilesName.VIS, "End at " + date_now, MessageType.HEADER)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('vis_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
