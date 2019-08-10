import os
import argparse


def _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path,
           val_data_folder_path, val_labels_folder_path, val_ids_path):
    # this must be imported after setting CUDA_VISIBLE_DEVICES environment variable, otherwise it won't work
    from train import run

    run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path,
           val_data_folder_path, val_labels_folder_path, val_ids_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current_experiment', help='Experiment name', required=True)
    parser.add_argument('--train_data_folder_path', required=True)
    parser.add_argument('--train_labels_folder_path', required=True)
    parser.add_argument('--train_ids_file_path', required=True)
    parser.add_argument('--val_data_folder_path', required=False)
    parser.add_argument('--val_labels_folder_path', required=False)
    parser.add_argument('--val_ids_file_path', required=True)
    parser.add_argument('--GPUs', required=False, type=str)

    args = parser.parse_args()
    current_experiment = args.current_experiment
    train_data_folder_path = args.train_data_folder_path
    train_labels_folder_path = args.train_labels_folder_path
    train_ids_path = args.train_ids_file_path
    val_data_folder_path = args.val_data_folder_path
    val_labels_folder_path = args.val_labels_folder_path
    val_ids_path = args.val_ids_file_path
    GPUs = args.GPUs

    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    if not val_data_folder_path and not val_labels_folder_path:
        # use the same folder from the training arguments
        val_data_folder_path = train_data_folder_path
        val_labels_folder_path = train_labels_folder_path

    _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path,
           val_data_folder_path, val_labels_folder_path, val_ids_path)


if __name__ == '__main__':
    main()
