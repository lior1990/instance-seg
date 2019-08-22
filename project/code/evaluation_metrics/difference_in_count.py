import numpy as np


def _different_in_count(prediction, ground_truth):
    """
    computes DiC score of a predicted label and ground-truth segmentation.
    :param prediction: (1, h, w) or (h, w) nd-array segmentation
    :param ground_truth: (1, h, w) or (h, w) nd-array ground-truth segmentation
    :return: DiC score
    """
    prediction_num_of_instances = len(np.unique(prediction))
    ground_truth_num_of_instances = len(np.unique(ground_truth))
    return prediction_num_of_instances - ground_truth_num_of_instances


def absolute_difference_in_count(prediction, ground_truth):
    """
    computes |DiC| score of a predicted label and ground-truth segmentation.
    :param prediction: (1, h, w) or (h, w) nd-array segmentation
    :param ground_truth: (1, h, w) or (h, w) nd-array ground-truth segmentation
    :return: |DiC| score
    """
    difference_in_count_score = _different_in_count(prediction, ground_truth)
    return np.abs(difference_in_count_score)
