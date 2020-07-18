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
from research.deeplab.datasets import data_generator

from sklearn.metrics import confusion_matrix

from datetime import datetime
from research.deeplab.utils.general_utils import *

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.
flags.DEFINE_list('inference_crop_size', '513,513', 'Crop size [height, width] for visualization.')

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

flags.DEFINE_string('dataset', 'pascal_voc_seg_58_parts', 'Name of the segmentation dataset.')

flags.DEFINE_string('inference_split', 'val', 'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('save_softmax', 0, 'Also save softmax predictions.')

flags.DEFINE_integer('model_softmax', 0, 'Use model_softmax or not (Default use model)')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print_message(FilesName.VIS, "Start at " + date_now, MessageType.HEADER)

    # Get dataset-dependent information.
    dataset = data_generator.Dataset(
        dataset_name=FLAGS.dataset,
        split_name=FLAGS.inference_split,
        dataset_dir=FLAGS.dataset_dir,
        batch_size=1,
        crop_size=[int(sz) for sz in FLAGS.inference_crop_size],
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        model_variant=FLAGS.model_variant,
        is_training=False,
        should_shuffle=False,
        should_repeat=False)

    with tf.Graph().as_default():
        samples = dataset.get_one_shot_iterator(use_softmax=1, use_gt_macro_classes=1, use_edges=1).get_next()

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_of_classes},
            crop_size=[int(sz) for sz in FLAGS.inference_crop_size],
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')

            if FLAGS.model_softmax == 1:
                predictions = model_softmax.predict_labels(samples[common.IMAGE],
                                                           samples[common.SOFTMAX],
                                                           model_options=model_options,
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

            else:
                predictions = model.predict_labels_multi_scale(samples[common.IMAGE],
                                                               model_options=model_options,
                                                               eval_scales=FLAGS.eval_scales,
                                                               add_flipped_images=FLAGS.add_flipped_images)

        predictions_argmax = predictions[common.OUTPUT_TYPE]

        checkpoints_iterator = contrib_training.checkpoints_iterator(FLAGS.checkpoint_dir,
                                                                     min_interval_secs=FLAGS.eval_interval_secs)

        mat = np.zeros(shape=(dataset.num_of_classes, dataset.num_of_classes), dtype=np.int32)

        for checkpoint_path in checkpoints_iterator:
            tf.logging.info('Starting inference at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))
            tf.logging.info('Visualizing with model %s', checkpoint_path)

            scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=scaffold,
                master=FLAGS.master,
                checkpoint_filename_with_path=checkpoint_path)

            with tf.train.MonitoredSession(session_creator=session_creator, hooks=None) as sess:
                number_of_images = data_generator._DATASETS_INFORMATION[FLAGS.dataset].splits_to_sizes[FLAGS.inference_split]

                for image_index in tqdm(range(number_of_images)):
                    (label,
                     prediction,
                     image_name,
                     image_height,
                     image_width) = sess.run([samples[common.LABEL],
                                             predictions_argmax,
                                             samples[common.IMAGE_NAME],
                                             samples[common.HEIGHT],
                                             samples[common.WIDTH]])

                    label = np.squeeze(label[0])
                    image_height = np.squeeze(image_height[0])
                    image_width = np.squeeze(image_width[0])
                    prediction = np.squeeze(prediction[0])
                    label = label[:image_height, :image_width]
                    prediction = prediction[:image_height, :image_width]

                    ignore_label_value = data_generator._DATASETS_INFORMATION[FLAGS.dataset].ignore_label

                    mask = label != ignore_label_value

                    label = label[mask]
                    prediction = prediction[mask]

                    tmp = confusion_matrix(label, prediction, range(dataset.num_of_classes))
                    mat = mat + tmp

                compute_and_print_IoU_per_class(confusion_matrix=mat,
                                                num_classes=dataset.num_of_macro_classes,
                                                num_parts=dataset.num_of_classes)


            tf.logging.info('Finished inference at ' + time.strftime('%Y-%m-%d-%H:%M:%S', time.gmtime()))

    date_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    print_message(FilesName.VIS, "End at " + date_now, MessageType.HEADER)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
