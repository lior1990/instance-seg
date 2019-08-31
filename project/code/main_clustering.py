import os
from utils.argument_parser import train_argument_parser


def _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path):
    # this must be imported after setting CUDA_VISIBLE_DEVICES environment variable, otherwise it won't work
    from trainClustering import run

    run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path)


def main():
    current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path, GPUs = train_argument_parser()

    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    print('experiment:', current_experiment)
    print('train data folder path:', train_data_folder_path)
    print('train labels folder path:', train_labels_folder_path)
    print('train ids path:', train_ids_path)

    _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path)


if __name__ == '__main__':
    main()
