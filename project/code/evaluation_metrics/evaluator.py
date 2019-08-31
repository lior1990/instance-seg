from evaluation_metrics.dice import symmetric_best_dice, foreground_background_dice
from evaluation_metrics.difference_in_count import absolute_difference_in_count


class Evaluator(object):
    def __init__(self):
        self._evaluations = [
            ("SBD", symmetric_best_dice),
            ("FBD", foreground_background_dice),
            ("|DiC|", absolute_difference_in_count)
        ]

        self._results = {evaluation_name: 0 for evaluation_name, evaluation_func in self._evaluations}
        self._calls_counter = 0

    def evaluate(self, prediction, ground_truth):
        """
        Evaluates the segmentation of prediction against ground-truth.
        :param prediction:
        :param ground_truth:
        """
        self._calls_counter += 1
        currEvalScores = {evaluation_name: 0 for evaluation_name, evaluation_func in self._evaluations}
        for evaluation_name, evaluation_func in self._evaluations:
            evaluation_score = evaluation_func(prediction, ground_truth)
            currEvalScores[evaluation_name] = evaluation_score
            self._results[evaluation_name] += evaluation_score
        return currEvalScores

    def get_average_results(self):
        return {key: value / self._calls_counter for key, value in self._results.items()}
