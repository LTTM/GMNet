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
"""Utility functions for training."""

import six
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework

from research.deeplab.core import preprocess_utils
from research.deeplab.core import utils

from research.deeplab.utils.general_utils import *


def _div_maybe_zero(total_loss, num_present):
    """Normalizes the total loss with the number of present pixels."""
    return tf.to_float(num_present > 0) * tf.math.divide(
        total_loss,
        tf.maximum(1e-5, num_present))


# Create num_classx58 mask subdivided into 21 blocks ->
def create_blocks_mask(num_classes, num_parts, one_zero=1):

    if one_zero == 1:
        blocks_mask = np.zeros([num_parts, num_parts], np.float32)
        map_pc = map_parts_to_classes(num_classes, num_parts)

        for class_indexes in map_pc:
            start_index = class_indexes[0]
            end_index = class_indexes[1]

            for row_index in range(start_index, end_index):
                for col_index in range(start_index, end_index):
                    blocks_mask[row_index, col_index] = 1.0

        blocks_mask_tensor = tf.convert_to_tensor(blocks_mask, dtype=tf.float32)

    else:
        blocks_mask = np.full([num_parts, num_parts], 0.2, np.float32)
        map_pc = map_parts_to_classes(num_classes, num_parts)

        for class_indexes in map_pc:
            start_index = class_indexes[0]
            end_index = class_indexes[1]

            for row_index in range(start_index, end_index):
                for col_index in range(start_index, end_index):
                    blocks_mask[row_index, col_index] = 0.8

        blocks_mask_tensor = tf.convert_to_tensor(blocks_mask, dtype=tf.float32)

    return blocks_mask_tensor


def compute_dilated_mask(image, ignore_label, class_index, dilation_kernel):
    # Mask without 255 values
    mask_ignore = tf.cast(tf.not_equal(image, ignore_label), dtype=tf.float32)

    # Mask of the specific class
    mask_class = tf.cast(tf.equal(image, class_index), dtype=tf.float32)

    # Final class mask
    mask = tf.multiply(mask_ignore, mask_class)

    # Dilation
    kernel = tf.zeros((dilation_kernel, dilation_kernel, 1))  # 2, 2, 1
    dilated_mask = tf.nn.dilation2d(mask, filter=kernel, strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding='SAME')

    return dilated_mask


# Compute the adjacency matrix of the classes for each batch of images:
# for each pair of two classes (background excluded) dilates the classes mask, multiplies them,
# in order to retrieve the adjacent pixels, and finally update the adjacency matrix according to the 'weighted' flag
def compute_adj_mat(image, adj_mat, num_classes, present_classes, ignore_label, dilation_kernel, weighted):

    num_present_classes = tf.shape(present_classes)[0]
    i = 1
    condition1 = lambda i, adj_mat: i < num_present_classes

    def body1(i, adj_mat):
        j = i + 1
        condition2 = lambda i, j, adj_mat: j < num_present_classes

        first_dilated_mask = compute_dilated_mask(image, ignore_label, present_classes[i], dilation_kernel)

        def body2(i, j, adj_mat):
            second_dilated_mask = compute_dilated_mask(image, ignore_label, present_classes[j], dilation_kernel)

            intersection = tf.multiply(first_dilated_mask, second_dilated_mask)

            adjacent_pixels = tf.cast(tf.count_nonzero(intersection), tf.int32)

            # WeightedAdjMat - The class1-class2 value contains the number of adjacent pixels if the 2 classes
            # are adjacent,  0 otherwise
            if weighted:
                indices = [[present_classes[i], present_classes[j], 0]]
                values = tf.reshape(adjacent_pixels, [1])
                shape = [num_classes, num_classes, 1]
                delta = tf.SparseTensor(indices, values, shape)
                adj_mat = adj_mat + tf.sparse_tensor_to_dense(delta)

            # SimpleAdjMat - The class1-class2 value contains 1 if the 2 classes are adjacent, 0 otherwise
            else:
                value = tf.cond(tf.greater(adjacent_pixels, 0), lambda: 1, lambda: 0)
                indices = [[present_classes[i], present_classes[j], 0]]
                values = tf.reshape(value, [1])
                shape = [num_classes, num_classes, 1]
                delta = tf.SparseTensor(indices, values, shape)
                adj_mat = adj_mat + tf.sparse_tensor_to_dense(delta)

            return i, j + 1, adj_mat

        i, j, adj_mat = tf.while_loop(condition2, body2, loop_vars=[i, j, adj_mat])
        return i + 1, adj_mat

    i, adj_mat = tf.while_loop(condition1, body1, loop_vars=[i, adj_mat])

    return tf.cast(tf.squeeze(adj_mat, -1), dtype=tf.float32)


def custom_error_l1(logits_adj_mat, labels_adj_mat, apply_block_mask=False, one_zero=1, num_classes=58, num_macro_classes=21):
    adj_predictions = tf.norm(tensor=logits_adj_mat, ord=1, axis=1)
    adj_labels = tf.norm(tensor=labels_adj_mat, ord=1, axis=1)

    diff = (adj_predictions - adj_labels)
    quad = diff * diff

    if apply_block_mask:
        block_mask = create_blocks_mask(num_macro_classes, num_classes, one_zero)
        quad = tf.math.multiply(quad, block_mask)

    loss = tf.reduce_mean(quad)

    return loss


def norm1_error(logits_adj_mat, labels_adj_mat, apply_block_mask=False, one_zero=1, num_classes=58, num_macro_classes=21):
    diff = (logits_adj_mat - labels_adj_mat)
    norm = tf.norm(tensor=diff, ord=1, axis=1)

    if apply_block_mask:
        block_mask = create_blocks_mask(num_macro_classes, num_classes, one_zero)
        norm = tf.math.multiply(norm, block_mask)

    loss = tf.reduce_mean(norm)

    return loss


def mean_squared_error(logits_adj_mat, labels_adj_mat, apply_block_mask=False, one_zero=1, num_classes=58,
                       num_macro_classes=21):
    diff = (logits_adj_mat - labels_adj_mat)
    norm = tf.norm(tensor=diff, ord=2, axis=1)
    square = norm * norm

    if apply_block_mask:
        block_mask = create_blocks_mask(num_macro_classes, num_classes, one_zero)
        square = tf.math.multiply(square, block_mask)

    loss = tf.reduce_mean(square)

    return loss


def kullback_leibler_divergence(logits_adj_mat, labels_adj_mat, apply_block_mask=False, one_zero=1, num_classes=58,
                                num_macro_classes=21):
    if logits_adj_mat is None:
        return 0

    epsilon = 0.00000001
    div = labels_adj_mat / logits_adj_mat
    kl_div = labels_adj_mat * tf.log(div + epsilon)

    if apply_block_mask:
        block_mask = create_blocks_mask(num_macro_classes, num_classes, one_zero)
        kl_div = tf.math.multiply(kl_div, block_mask)

    loss = tf.reduce_mean(kl_div)

    return loss


def add_edge_loss(pred_edges,
                  edges,
                  edges_lambda_loss,
                  scope=None,):
    if edges is None:
        raise ValueError('No label for edges loss.')

    loss_scope = None
    if scope:
        loss_scope = scope

    default_loss_scope = 'edges_loss'

    pred_edges = tf.image.resize_nearest_neighbor(pred_edges,
                                                  preprocess_utils.resolve_shape(edges, 4)[1:3],
                                                  align_corners=True)

    all_one_mask = tf.ones(tf.shape(edges), tf.float32)
    tot_num = tf.reduce_sum(tf.math.count_nonzero(all_one_mask, axis=-1))
    pos_num = tf.reduce_sum(tf.math.count_nonzero(edges, axis=-1))
    neg_num = tot_num - pos_num

    weight_pos = neg_num / (pos_num + neg_num)
    weight_neg = pos_num / (pos_num + neg_num)
    class_weights = [weight_neg, weight_pos]

    sample_weights = tf.gather(class_weights, edges)

    pred_edges = tf.cast(pred_edges, tf.float32)

    edges = tf.reshape(edges, shape=[-1])
    edges = tf.one_hot(edges, 2, on_value=1.0, off_value=0.0)

    pred_edges = tf.reshape(pred_edges, shape=[-1, 2])

    sample_weights = tf.reshape(sample_weights, shape=[-1])

    loss = tf.losses.softmax_cross_entropy(edges, pred_edges, weights=sample_weights, loss_collection=None)

    loss = edges_lambda_loss * loss

    loss_type = 'edges_loss'

    loss = tf.identity(loss, name=loss_type)

    with tf.name_scope(loss_scope, default_loss_scope, [pred_edges, edges]):
        # Compute the adjacency loss
        tf.losses.add_loss(loss)


def add_spectral_loss(scales_to_logits,
                      labels,
                      num_classes,
                      ignore_label,
                      weighted,
                      class_dilation_kernel=2,
                      eigen_lambda_loss=1,
                      keep_max=3,
                      use_eigen_vectors=1,
                      scope=None):
    """Adds spectral adjacent matrix loss for logits of each scale."""

    if labels is None:
        raise ValueError('No label for adjacency graph loss.')

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        default_loss_scope = 'argmax_adjacency_loss'

        logits = tf.image.resize_bilinear(logits,
                                          preprocess_utils.resolve_shape(labels, 4)[1:3],
                                          align_corners=True)

        logits = tf.cast(tf.argmax(logits, 3), dtype=tf.int32)
        logits = tf.expand_dims(logits, -1)

        reshaped1 = tf.reshape(logits, [-1])
        reshaped2 = tf.reshape(labels, [-1])
        concat = tf.concat([reshaped1, reshaped2], 0)
        unique = tf.unique(concat)
        sorted_classes = tf.sort(unique[0])
        keep_mask = tf.cast(tf.not_equal(sorted_classes, ignore_label), dtype=tf.int32)
        sorted_classes = tf.multiply(sorted_classes, keep_mask)

        logits_adj_mat = tf.zeros([num_classes, num_classes, 1], tf.int32)
        labels_adj_mat = tf.zeros([num_classes, num_classes, 1], tf.int32)

        logits_adj_mat = compute_adj_mat(image=logits,
                                         adj_mat=logits_adj_mat,
                                         num_classes=num_classes,
                                         present_classes=sorted_classes,
                                         ignore_label=ignore_label,
                                         dilation_kernel=class_dilation_kernel,
                                         weighted=weighted)

        labels_adj_mat = compute_adj_mat(image=labels,
                                         adj_mat=labels_adj_mat,
                                         num_classes=num_classes,
                                         present_classes=sorted_classes,
                                         ignore_label=ignore_label,
                                         dilation_kernel=class_dilation_kernel,
                                         weighted=weighted)

        if weighted:
            logits_adj_mat = tf.math.l2_normalize(logits_adj_mat, axis=0)
            labels_adj_mat = tf.math.l2_normalize(labels_adj_mat, axis=0)

        if logits_adj_mat is None:
            return 0

        full_logits_adj_mat = logits_adj_mat + tf.transpose(logits_adj_mat)
        full_labels_adj_mat = labels_adj_mat + tf.transpose(labels_adj_mat)

        eigen_values_logits, eigen_vectors_logits = tf.linalg.eigh(full_logits_adj_mat)
        eigen_values_labels, eigen_vectors_labels = tf.linalg.eigh(full_labels_adj_mat)

        max_values_logits, max_indices_logits = tf.math.top_k(eigen_values_logits, keep_max, sorted=True)
        max_values_labels, max_indices_labels = tf.math.top_k(eigen_values_labels, keep_max, sorted=True)

        max_vec_logits_list = []
        max_vec_labels_list = []

        for num in range(0, keep_max):
            max_vec_logits_list.append(eigen_vectors_logits[max_indices_logits[num]])
            max_vec_labels_list.append(eigen_vectors_labels[max_indices_logits[num]])

        max_eigen_vectors_logits = tf.concat(max_vec_logits_list, 0)
        max_eigen_vectors_labels = tf.concat(max_vec_labels_list, 0)

        if use_eigen_vectors:
            diff = max_eigen_vectors_logits - max_eigen_vectors_labels
        else:
            diff = max_values_logits - max_values_labels

        norm = tf.norm(tensor=diff, ord=2, axis=None)

        loss = tf.reduce_mean(norm)

        loss = eigen_lambda_loss * loss

        loss_type = 'spectral_loss'

        loss = tf.identity(loss, name=loss_type)

        with tf.name_scope(loss_scope, default_loss_scope, [logits, labels]):
            # Compute the adjacency loss
            tf.losses.add_loss(loss)


def add_adjacent_graphs_loss(scales_to_logits,
                             labels,
                             num_classes,
                             num_macro_classes,
                             apply_block_mask,
                             one_zero_mask,
                             ignore_label,
                             weighted,
                             lambda_loss=1.0,
                             loss_type="custom_error",
                             class_dilation_kernel=2,
                             scope=None):
    """Adds adjacent graphs loss for logits of each scale."""

    if labels is None:
        raise ValueError('No label for adjacency graph loss.')

    if loss_type not in ['custom_error', 'norm1_error', 'mean_squared_error', 'kullback_leibler_divergence']:
        raise ValueError('Loss type not supported.')

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        default_loss_scope = 'argmax_adjacency_loss'

        logits = tf.image.resize_bilinear(logits,
                                          preprocess_utils.resolve_shape(labels, 4)[1:3],
                                          align_corners=True)

        logits = tf.cast(tf.argmax(logits, 3), dtype=tf.int32)
        logits = tf.expand_dims(logits, -1)

        reshaped1 = tf.reshape(logits, [-1])
        reshaped2 = tf.reshape(labels, [-1])
        concat = tf.concat([reshaped1, reshaped2], 0)
        keep_mask = tf.cast(tf.not_equal(concat, ignore_label), dtype=tf.int32)
        concat = tf.multiply(concat, keep_mask)
        unique = tf.unique(concat)
        sorted_classes = tf.sort(unique[0])


        logits_adj_mat = tf.zeros([num_classes, num_classes, 1], tf.int32)
        labels_adj_mat = tf.zeros([num_classes, num_classes, 1], tf.int32)

        logits_adj_mat = compute_adj_mat(image=logits,
                                         adj_mat=logits_adj_mat,
                                         num_classes=num_classes,
                                         present_classes=sorted_classes,
                                         ignore_label=ignore_label,
                                         dilation_kernel=class_dilation_kernel,
                                         weighted=weighted)

        labels_adj_mat = compute_adj_mat(image=labels,
                                         adj_mat=labels_adj_mat,
                                         num_classes=num_classes,
                                         present_classes=sorted_classes,
                                         ignore_label=ignore_label,
                                         dilation_kernel=class_dilation_kernel,
                                         weighted=weighted)

        if weighted:
            logits_adj_mat = tf.math.l2_normalize(logits_adj_mat, axis=0)
            labels_adj_mat = tf.math.l2_normalize(labels_adj_mat, axis=0)

        if logits_adj_mat is None:
            return 0

        loss = 0
        if loss_type == 'custom_error':
            loss = custom_error_l1(logits_adj_mat,
                                   labels_adj_mat,
                                   apply_block_mask=apply_block_mask,
                                   one_zero=one_zero_mask,
                                   num_classes=num_classes,
                                   num_macro_classes=num_macro_classes)

        elif loss_type == 'norm1_error':
            loss = norm1_error(logits_adj_mat,
                               labels_adj_mat,
                               apply_block_mask=apply_block_mask,
                               one_zero=one_zero_mask,
                               num_classes=num_classes,
                               num_macro_classes=num_macro_classes)

        elif loss_type == 'mean_squared_error':
            loss = mean_squared_error(logits_adj_mat,
                                      labels_adj_mat,
                                      apply_block_mask=apply_block_mask,
                                      one_zero=one_zero_mask,
                                      num_classes=num_classes,
                                      num_macro_classes=num_macro_classes)

        elif loss_type == 'kullback_leibler_divergence':
            loss = kullback_leibler_divergence(logits_adj_mat,
                                               labels_adj_mat,
                                               apply_block_mask=apply_block_mask,
                                               one_zero=one_zero_mask,
                                               num_classes=num_classes,
                                               num_macro_classes=num_macro_classes)

        loss = lambda_loss * loss

        loss = tf.identity(loss, name=loss_type)

        with tf.name_scope(loss_scope, default_loss_scope, [logits, labels]):
            # Compute the adjacency loss
            tf.losses.add_loss(loss)


def add_aux_loss(scales_to_logits,
                 labels_macro_classes,
                 num_classes,
                 num_macro_classes,
                 ignore_label,
                 lambda_loss=1.0,
                 loss_weight=1.0,
                 upsample_logits=True,
                 hard_example_mining_step=0,
                 top_k_percent_pixels=1.0,
                 gt_is_matting_map=False,
                 scope=None):
    if labels_macro_classes is None:
        raise ValueError('No macro classes label for auxiliary softmax cross entropy loss.')

    # If input groundtruth is a matting map of confidence, check if the input
    # labels are floating point values.
    if gt_is_matting_map and not labels_macro_classes.dtype.is_floating:
        raise ValueError('Labels must be floats if groundtruth is a matting map.')

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(logits,
                                              preprocess_utils.resolve_shape(labels_macro_classes, 4)[1:3],
                                              align_corners=True)
            scaled_labels_macro_classes = labels_macro_classes

        else:
            if gt_is_matting_map:
                tf.logging.warning('Label downsampling with nearest neighbor may introduce artifacts.')

            scaled_labels_macro_classes = tf.image.resize_nearest_neighbor(
                labels_macro_classes,
                preprocess_utils.resolve_shape(logits, 4)[1:3],
                align_corners=True)

        # Transform logits from X parts to Y macro classes (object-level)
        map_pc = map_parts_to_classes(num_macro_classes, num_classes)

        map_pc_splitting = []
        for el in map_pc:
            map_pc_splitting.append(el[1] - el[0])

        # map_pc_splitting = [1,5,2,4,1,2,3,5,4,1,4,1,4,4,2,6,2,3,1,1,2]

        macro_class_logits = tf.split(logits, map_pc_splitting, axis=-1)

        for i in range(len(macro_class_logits)):
            macro_class_logits[i] = tf.reduce_sum(macro_class_logits[i], axis=-1)

        final_tensor = tf.stack(macro_class_logits, axis=-1)

        ###############################################

        scaled_labels_macro_classes = tf.reshape(scaled_labels_macro_classes, shape=[-1])
        weights = utils.get_label_weight_mask(scaled_labels_macro_classes, ignore_label,
                                              num_macro_classes, label_weights=loss_weight)
        # Dimension of keep_mask is equal to the total number of pixels.
        keep_mask = tf.cast(tf.not_equal(scaled_labels_macro_classes, ignore_label), dtype=tf.float32)

        train_labels = None
        final_tensor = tf.reshape(final_tensor, shape=[-1, num_macro_classes])

        if gt_is_matting_map:
            # When the groundtruth is integer label mask, we can assign class
            # dependent label weights to the loss. When the groundtruth is image
            # matting confidence, we do not apply class-dependent label weight (i.e.,
            # label_weight = 1.0).
            if loss_weight != 1.0:
                raise ValueError('loss_weight must equal to 1 if groundtruth is matting map.')

            # Assign label value 0 to ignore pixels. The exact label value of ignore
            # pixel does not matter, because those ignore_value pixel losses will be
            # multiplied to 0 weight.
            train_labels = scaled_labels_macro_classes * keep_mask

            train_labels = tf.expand_dims(train_labels, 1)
            train_labels = tf.concat([1 - train_labels, train_labels], axis=1)

        else:
            train_labels = tf.one_hot(scaled_labels_macro_classes, num_macro_classes, on_value=1.0, off_value=0.0)

        default_loss_scope = ('softmax_all_pixel_loss_aux'
                              if top_k_percent_pixels == 1.0 else
                              'softmax_hard_example_mining_aux')

        with tf.name_scope(loss_scope, default_loss_scope, [final_tensor, train_labels, weights]):
            # Compute the loss for all pixels.
            pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(train_labels, name='train_labels_stop_gradient_aux'),
                logits=final_tensor,
                name='pixel_losses_aux')

            weighted_pixel_losses = tf.multiply(pixel_losses, weights)

            if top_k_percent_pixels == 1.0:
                total_loss = tf.reduce_sum(weighted_pixel_losses)
                num_present = tf.reduce_sum(keep_mask)
                loss = _div_maybe_zero(total_loss, num_present)

            else:
                num_pixels = tf.to_float(tf.shape(final_tensor)[0])
                # Compute the top_k_percent pixels based on current training step.
                if hard_example_mining_step == 0:
                    # Directly focus on the top_k pixels.
                    top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)

                else:
                    # Gradually reduce the mining percent to top_k_percent_pixels.
                    global_step = tf.to_float(tf.train.get_or_create_global_step())
                    ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
                    top_k_pixels = tf.to_int32((ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)

                top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                              k=top_k_pixels,
                                              sorted=True,
                                              name='top_k_percent_pixels_aux')
                total_loss = tf.reduce_sum(top_k_losses)
                num_present = tf.reduce_sum(tf.to_float(tf.not_equal(top_k_losses, 0.0)))
                loss = _div_maybe_zero(total_loss, num_present)

            loss = lambda_loss * loss

            tf.losses.add_loss(loss)


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  lambda_loss=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  gt_is_matting_map=False,
                                                  scope=None):
    """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: A float or a list of loss weights. If it is a float, it means
      all the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, loss_weight = [0.1, 0.5] means the weight for label 0 is 0.1 and
      the weight for label 1 is 0.5.
    lambda_loss:
    upsample_logits: Boolean, upsample logits or not.
    hard_example_mining_step: An integer, the training step in which the hard
      exampling mining kicks off. Note that we gradually reduce the mining
      percent to the top_k_percent_pixels. For example, if
      hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
      mining percent will gradually reduce from 100% to 25% until 100K steps
      after which we only mine top 25% pixels.
    top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
      < 1.0, only compute the loss for the top k percent pixels (e.g., the top
      20% pixels). This is useful for hard pixel mining.
    gt_is_matting_map: If true, the groundtruth is a matting map of confidence
      score. If false, the groundtruth is an integer valued class mask.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None, or groundtruth is matting map while
      label is not floating value.
  """
    if labels is None:
        raise ValueError('No label for softmax cross entropy loss.')

    # If input groundtruth is a matting map of confidence, check if the input
    # labels are floating point values.
    if gt_is_matting_map and not labels.dtype.is_floating:
        raise ValueError('Labels must be floats if groundtruth is a matting map.')

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(logits,
                                              preprocess_utils.resolve_shape(labels, 4)[1:3],
                                              align_corners=True)
            scaled_labels = labels

        else:
            # Label is downsampled to the same size as logits.
            # When gt_is_matting_map = true, label downsampling with nearest neighbor
            # method may introduce artifacts. However, to avoid ignore_label from
            # being interpolated with other labels, we still perform nearest neighbor
            # interpolation.
            # TODO(huizhongc): Change to bilinear interpolation by processing padded
            # and non-padded label separately.
            if gt_is_matting_map:
                tf.logging.warning('Label downsampling with nearest neighbor may introduce artifacts.')

            scaled_labels = tf.image.resize_nearest_neighbor(
                labels,
                preprocess_utils.resolve_shape(logits, 4)[1:3],
                align_corners=True)

        scaled_labels = tf.reshape(scaled_labels, shape=[-1])
        weights = utils.get_label_weight_mask(scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
        # Dimension of keep_mask is equal to the total number of pixels.
        keep_mask = tf.cast(tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)

        train_labels = None
        logits = tf.reshape(logits, shape=[-1, num_classes])

        if gt_is_matting_map:
            # When the groundtruth is integer label mask, we can assign class
            # dependent label weights to the loss. When the groundtruth is image
            # matting confidence, we do not apply class-dependent label weight (i.e.,
            # label_weight = 1.0).
            if loss_weight != 1.0:
                raise ValueError('loss_weight must equal to 1 if groundtruth is matting map.')

            # Assign label value 0 to ignore pixels. The exact label value of ignore
            # pixel does not matter, because those ignore_value pixel losses will be
            # multiplied to 0 weight.
            train_labels = scaled_labels * keep_mask

            train_labels = tf.expand_dims(train_labels, 1)
            train_labels = tf.concat([1 - train_labels, train_labels], axis=1)

        else:
            train_labels = tf.one_hot(scaled_labels, num_classes, on_value=1.0, off_value=0.0)

        default_loss_scope = ('softmax_all_pixel_loss'
                              if top_k_percent_pixels == 1.0 else
                              'softmax_hard_example_mining')

        with tf.name_scope(loss_scope, default_loss_scope, [logits, train_labels, weights]):
            # Compute the loss for all pixels.
            pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(train_labels, name='train_labels_stop_gradient'),
                logits=logits,
                name='pixel_losses')

            weighted_pixel_losses = tf.multiply(pixel_losses, weights)

            if top_k_percent_pixels == 1.0:
                total_loss = tf.reduce_sum(weighted_pixel_losses)
                num_present = tf.reduce_sum(keep_mask)
                loss = _div_maybe_zero(total_loss, num_present)

            else:
                num_pixels = tf.to_float(tf.shape(logits)[0])
                # Compute the top_k_percent pixels based on current training step.
                if hard_example_mining_step == 0:
                    # Directly focus on the top_k pixels.
                    top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)

                else:
                    # Gradually reduce the mining percent to top_k_percent_pixels.
                    global_step = tf.to_float(tf.train.get_or_create_global_step())
                    ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
                    top_k_pixels = tf.to_int32((ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)

                top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                              k=top_k_pixels,
                                              sorted=True,
                                              name='top_k_percent_pixels')
                total_loss = tf.reduce_sum(top_k_losses)
                num_present = tf.reduce_sum(tf.to_float(tf.not_equal(top_k_losses, 0.0)))
                loss = _div_maybe_zero(total_loss, num_present)

            loss = lambda_loss * loss

            tf.losses.add_loss(loss)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
    """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
    if tf_initial_checkpoint is None:
        tf.logging.info('Not initializing the model from a checkpoint.')
        return None

    if tf.train.latest_checkpoint(train_logdir):
        tf.logging.info('Ignoring initialization; other checkpoint exists')
        return None

    tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

    # Variables that will not be restored.
    exclude_list = ['global_step']
    if not initialize_last_layer:
        exclude_list.extend(last_layers)

    variables_to_restore = contrib_framework.get_variables_to_restore(exclude=exclude_list)

    if variables_to_restore:
        init_op, init_feed_dict = contrib_framework.assign_from_checkpoint(
            tf_initial_checkpoint,
            variables_to_restore,
            ignore_missing_vars=ignore_missing_vars)
        global_step = tf.train.get_or_create_global_step()

        def restore_fn(sess):
            sess.run(init_op, init_feed_dict)
            sess.run([global_step])

        return restore_fn

    return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
    """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
    gradient_multipliers = {}

    for var in tf.model_variables():
        # Double the learning rate for biases.
        if 'biases' in var.op.name:
            gradient_multipliers[var.op.name] = 2.

        # Use larger learning rate for last layer variables.
        for layer in last_layers:
            if layer in var.op.name and 'biases' in var.op.name:
                gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
                break
            elif layer in var.op.name:
                gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
                break

    return gradient_multipliers


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none',
                            decay_steps=0.0,
                            end_learning_rate=0.0,
                            boundaries=None,
                            boundary_learning_rates=None):
    """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.
    decay_steps: Float, `decay_steps` for polynomial learning rate.
    end_learning_rate: Float, `end_learning_rate` for polynomial learning rate.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries.
    boundary_learning_rates: A list of `Tensor`s or `float`s or `int`s that
      specifies the values for the intervals defined by `boundaries`. It should
      have one more element than `boundaries`, and all elements should have the
      same type.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
    ValueError: If `boundaries` and `boundary_learning_rates` are not set for
      multi_steps learning rate decay.
  """
    global_step = tf.train.get_or_create_global_step()
    adjusted_global_step = tf.maximum(global_step - slow_start_step, 0)
    if decay_steps == 0.0:
        tf.logging.info('Setting decay_steps to total training steps.')
        decay_steps = training_number_of_steps - slow_start_step
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            adjusted_global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
            base_learning_rate,
            adjusted_global_step,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=learning_power)
    elif learning_policy == 'cosine':
        learning_rate = tf.train.cosine_decay(
            base_learning_rate,
            adjusted_global_step,
            training_number_of_steps - slow_start_step)
    elif learning_policy == 'multi_steps':
        if boundaries is None or boundary_learning_rates is None:
            raise ValueError('Must set `boundaries` and `boundary_learning_rates` '
                             'for multi_steps learning rate decay.')
        learning_rate = tf.train.piecewise_constant_decay(
            adjusted_global_step,
            boundaries,
            boundary_learning_rates)
    else:
        raise ValueError('Unknown learning policy.')

    adjusted_slow_start_learning_rate = slow_start_learning_rate
    if slow_start_burnin_type == 'linear':
        # Do linear burnin. Increase linearly from slow_start_learning_rate and
        # reach base_learning_rate after (global_step >= slow_start_steps).
        adjusted_slow_start_learning_rate = (
                slow_start_learning_rate +
                (base_learning_rate - slow_start_learning_rate) *
                tf.to_float(global_step) / slow_start_step)
    elif slow_start_burnin_type != 'none':
        raise ValueError('Unknown burnin type.')

    # Employ small learning rate at the first few steps for warm start.
    return tf.where(global_step < slow_start_step, adjusted_slow_start_learning_rate, learning_rate)
