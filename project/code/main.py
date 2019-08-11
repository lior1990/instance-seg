import os
import argparse
from datetime import datetime


# def _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path,
#            val_data_folder_path, val_labels_folder_path, val_ids_path):
#     # this must be imported after setting CUDA_VISIBLE_DEVICES environment variable, otherwise it won't work
#     from train import run
#
#     run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path,
#         val_data_folder_path, val_labels_folder_path, val_ids_path)
def _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path):
    # this must be imported after setting CUDA_VISIBLE_DEVICES environment variable, otherwise it won't work
    from train import run

    run(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path)


def main():
    default_experiment_name = 'exp_' + str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    default_train_data_path = os.path.join('..', '..', 'COCO', 'train2017', '')
    default_train_labels_path = os.path.join('..', '..', 'COCO', 'train2017labels', 'instance_labels', '')
    # default_val_data_path = os.path.join('..', '..', 'COCO', 'val2017', '')
    # default_val_labels_path = os.path.join('..', '..', 'COCO', 'val2017labels', 'instance_labels', '')
    default_train_ids_file = os.path.join('..', '..', 'COCO', 'train2017labels', 'images_ids.txt')
    # default_val_ids_file = os.path.join('..', '..', 'COCO', 'val2017labels', 'images_ids.txt')

    parser = argparse.ArgumentParser()
    parser.add_argument('--current_experiment', help='Experiment name', required=False, default=default_experiment_name)
    parser.add_argument('--train_data_folder_path', required=False, default=default_train_data_path)
    parser.add_argument('--train_labels_folder_path', required=False, default=default_train_labels_path)
    # parser.add_argument('--validation_data_folder_path', required=False, default=default_val_data_path)
    # parser.add_argument('--validation_labels_folder_path', required=False, default=default_val_labels_path)
    parser.add_argument('--train_ids_file_path', required=False, default=default_train_ids_file)
    # parser.add_argument('--val_ids_file_path', required=False, default=default_val_ids_file)
    parser.add_argument('--GPUs', required=False, type=str)

    args = parser.parse_args()
    current_experiment = args.current_experiment
    train_data_folder_path = args.train_data_folder_path
    train_labels_folder_path = args.train_labels_folder_path
    train_ids_path = args.train_ids_file_path
    # val_data_folder_path = args.validation_data_folder_path
    # val_labels_folder_path = args.validation_labels_folder_path
    # val_ids_path = args.val_ids_file_path
    GPUs = args.GPUs

    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs


    # if not val_data_folder_path and not val_labels_folder_path:
    #     # use the same folder from the training arguments
    #     val_data_folder_path = train_data_folder_path
    #     val_labels_folder_path = train_labels_folder_path


    print('experiment:',current_experiment)
    print('train data folder path:',train_data_folder_path)
    print('train labels folder path:',train_labels_folder_path)
    print('train ids path:',train_ids_path)
    # print('val data folder path:',val_data_folder_path)
    # print('val labels folder path:',val_labels_folder_path)
    # print('val ids path:',val_ids_path)
    # _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path,
    #        val_data_folder_path, val_labels_folder_path, val_ids_path)
    _train(current_experiment, train_data_folder_path, train_labels_folder_path, train_ids_path)


if __name__ == '__main__':
    main()
