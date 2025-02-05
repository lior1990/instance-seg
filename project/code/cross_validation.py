import os
import pickle

from config import feature_extraction_chkpts_dir as chkpts_dir
from config import config_logger
from loss_params import LossParams
from utils.objects import DataSetParams

EXECUTION_ORDER = "execution_order"
EXECUTION_COUNTER = "execution_counter"
BEST_EXPERIMENT_INDEX = "best_experiment_index"
MIN_LOSS = "min_loss"

METADATA_NAME = "cv_metadata.pkl"


class CrossValidation(object):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.params_search = \
            {"dv": [2],
             "dd": [20],
             "include_edges": [False, True],
             "include_weighted_mean": [False, True],
             "alpha": [2]
             }
        # the keys of this dict represents a feature params, that without being ON, their dict values has no meaning
        self.features_params = {"include_edges": ["edges_max_pixels"],
                                "include_weighted_mean": ["center_weight", "corners_weight"]
                                }
        self._build_checkpoints_dir()
        self.logger = config_logger(self.experiment_name)
        metadata = self.load_experiment()

        if metadata:
            self.execution_order = metadata[EXECUTION_ORDER]
            self.execution_counter = metadata[EXECUTION_COUNTER]
            self.best_experiment_index = metadata[BEST_EXPERIMENT_INDEX]
            self.min_loss = metadata[MIN_LOSS]
            self.logger.info("Starting experiment %s from %s out of %s", self.experiment_name,
                             self.execution_counter, len(self.execution_order))
        else:
            self.execution_order = self._build_execution_order()
            self.execution_counter = 0
            self.best_experiment_index = None
            self.min_loss = None
            self.logger.info("Starting experiment %s from scratch", self.experiment_name)

    def run(self, train_data_folder_path, train_labels_folder_path, train_ids_path):

        train_data_set_params = DataSetParams(train_data_folder_path, train_labels_folder_path, train_ids_path)

        for i in range(self.execution_counter, len(self.execution_order)):
            current_params = self.execution_order[i]
            sub_experiment_name = self._build_sub_experiment_name(current_params)

            self.logger.info("Working on params: %s for experiment %s (%s out of %s)", current_params,
                             self.experiment_name, i, len(self.execution_order))

            loss = self._execute(current_params, sub_experiment_name, train_data_set_params)

            self.logger.info("experiment %s got loss: %s with params: %s", self.experiment_name, loss, current_params)

            if self.min_loss is None or loss < self.min_loss:
                self.min_loss = loss
                self.best_experiment_index = i
                self.logger.info("experiment %s with params: %s got the lowest loss %s!", self.experiment_name,
                                 current_params, loss)

            self.execution_counter = i + 1
            self._save()

    def _execute(self, current_params, sub_experiment_name, train_data_set_params):
        from train import run
        loss_params = LossParams(**current_params)
        loss = run(self.experiment_name, train_data_set_params, loss_params, sub_experiment_name=sub_experiment_name)
        return loss

    def _save(self):
        metadata = {EXECUTION_ORDER: self.execution_order,
                    EXECUTION_COUNTER: self.execution_counter,
                    BEST_EXPERIMENT_INDEX: self.best_experiment_index,
                    MIN_LOSS: self.min_loss}
        pickle.dump(metadata,
                    open(os.path.join(os.path.join(chkpts_dir, self.experiment_name, METADATA_NAME)), "wb")
                    )

        self.logger.info("Saved %s", self.experiment_name)

    def _build_execution_order(self):
        parameters_cartesian_product = self._cartesian_product(*[[{k: v} for v in self.params_search[k]] for k in self.params_search])

        execution_order = list()
        sub_experiment_names = list()

        for pcp in parameters_cartesian_product:
            combined_dict = dict()
            for item in pcp:
                combined_dict.update(item)

            combined_dict = self._drop_irrelevant_params(combined_dict)
            sub_exp_name = self._build_sub_experiment_name(combined_dict)

            if sub_exp_name not in sub_experiment_names:
                execution_order.append(combined_dict)
                sub_experiment_names.append(sub_exp_name)

        return execution_order

    def _drop_irrelevant_params(self, combined_dict):
        """
        This function drops irrelevant params from execution.
        For example, if "include_edges" is False, and "edges_max_pixels" has [100, 200] values,
        there's no point running an experiment with (False, 100) and (False, 200)
        :param combined_dict:
        :return:
        """
        for feature_param, dependent_params_list in self.features_params.items():
            if feature_param in combined_dict:
                if not combined_dict[feature_param]:
                    for dependent_param in dependent_params_list:
                        if dependent_param in combined_dict:
                            combined_dict.pop(dependent_param)
                            self.logger.debug("Dropped param: %s since %s is False!", dependent_param, feature_param)

        return combined_dict

    def load_experiment(self):
        metadata_path = os.path.join(os.path.join(chkpts_dir, self.experiment_name, METADATA_NAME))
        metadata = None

        if os.path.exists(metadata_path):
            self.logger.info("Loading metadata for %s" % self.experiment_name)
            metadata = pickle.load(open(metadata_path, "rb"))

        return metadata

    @staticmethod
    def _build_sub_experiment_name(current_params):
        sorted_keys = sorted(current_params.keys())
        name = ""

        for key in sorted_keys:
            value = str(current_params[key])
            value = value.replace('.', '__')  # replace '.' to avoid logger issues
            name += "%s%s_" % (key, value)

        return name[:-1]  # crop the last _

    def _build_checkpoints_dir(self):
        try:
            os.makedirs(os.path.join(chkpts_dir, self.experiment_name))
        except:
            pass

    def _cartesian_product(self, *X):
        if len(X) == 1: #special case, only X1
            return [ (x0,) for x0 in X[0] ]
        else:
            return [(x0,) + t1 for x0 in X[0] for t1 in self._cartesian_product(*X[1:])]
