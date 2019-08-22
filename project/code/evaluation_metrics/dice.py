import numpy as np


def symmetric_best_dice(prediction, ground_truth):
    """
    computes SBD of a predicted label and ground-truth segmentation.
    :param prediction: (1, h, w) or (h, w) nd-array segmentation
    :param ground_truth: (1, h, w) or (h, w) nd-array ground-truth segmentation
    :return: SBD score
    """
    score1 = _best_dice_score(prediction, ground_truth)
    score2 = _best_dice_score(ground_truth, prediction)
    return min([score1, score2])


def foreground_background_dice(prediction, ground_truth):
    """
    computes FBD of a predicted label and ground-truth segmentation.
    :param prediction: (1, h, w) or (h, w) nd-array segmentation
    :param ground_truth: (1, h, w) or (h, w) nd-array ground-truth segmentation
    :return: FBD score
    """
    _, prediction_background_instance = _get_instances_and_bg_instance(prediction)
    _, ground_truth_background_instance = _get_instances_and_bg_instance(ground_truth)

    prediction_mask = np.where(prediction != prediction_background_instance, 1, 0)
    ground_truth_mask = np.where(ground_truth != ground_truth_background_instance, 1, 0)

    return _dice_score(prediction_mask, ground_truth_mask)


def _best_dice_score(x, y):
    """
    computes Best Dice of a predicted label and ground-truth segmentation. this is done for
    OBJECTS (no background!) with no regard to classes.
    :param x: (1, h, w) or (h, w) nd-array
    :param y: (1, h, w) or (h, w) nd-array
    :return: Best Dice score
    """
    x_instances_without_bg, _ = _get_instances_and_bg_instance(x)
    y_instances_without_bg, _ = _get_instances_and_bg_instance(y)

    total_score = 0

    for x_instance in x_instances_without_bg:
        max_val = 0
        for y_instance in y_instances_without_bg:
            x_mask = np.where(x == x_instance, 1, 0)
            y_mask = np.where(y == y_instance, 1, 0)

            score = _dice_score(x_mask, y_mask)

            max_val = max([max_val, score])

        total_score += max_val

    return total_score/len(x_instances_without_bg)


def _dice_score(x_mask, y_mask):
    overlap = np.sum(np.logical_and(x_mask, y_mask))
    score = 2.0 * overlap / np.sum(x_mask + y_mask)
    return score


def _get_instances_and_bg_instance(x):
    """
    :return: Returns a tuple of size 2 where the first item is the list of instances in x,
    and the second is the background instance
    """
    x_instances_with_bg, x_counts = np.unique(x, return_counts=True)
    x_background_instance = x_instances_with_bg[np.argmax(x_counts)]
    x_instances = list(filter(lambda instance: instance != x_background_instance, x_instances_with_bg))
    return x_instances, x_background_instance
