import os

from utils.argument_parser import train_argument_parser


def _train(current_experiment, train_data_set_params, val_data_set_params):
    # this must be imported after setting CUDA_VISIBLE_DEVICES environment variable, otherwise it won't work
    from cross_validation import CrossValidation
    cv = CrossValidation(current_experiment)
    cv.run(train_data_set_params, val_data_set_params)


def main():
    current_experiment, train_data_set_params, val_data_set_params, GPUs = train_argument_parser()

    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    print('experiment:', current_experiment)
    print('train data folder path:', train_data_set_params.data_folder_path)
    print('train labels folder path:', train_data_set_params.labels_folder_path)
    print('train ids path:', train_data_set_params.ids_path)
    print('val data folder path:', val_data_set_params.data_folder_path)
    print('val labels folder path:', val_data_set_params.labels_folder_path)
    print('val ids path:', val_data_set_params.ids_path)

    _train(current_experiment, train_data_set_params, val_data_set_params)


if __name__ == '__main__':
    main()
