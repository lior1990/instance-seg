from torch.autograd import Variable

from costum_dataset import CostumeDataset
from utils.argument_parser import validation_argument_parser
from prediction import predict_label
from config import *
from evaluation_metrics.evaluator import Evaluator
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(model, dataloader):
    '''
    evaluates average loss of a model and average metrics as defined in Evaluator class.
    :param model: the model to use for evaluation_metrics
    :param dataloader: a dataloader with the validation set
    :return: average loss, average_evaluation_results (dict)
    '''
    evaluator = Evaluator()

    # running_loss = 0

    for i, batch in enumerate(dataloader):
        inputs = Variable(batch['image'].type(float_type))
        labels = batch['label'].cpu().numpy()
        # label_edges = batch['labelEdges'].cpu().numpy()

        results = model(inputs, None, None)
        features = results[0]
        # losses = model.loss(features, labels, label_edges)
        # current_loss = losses.mean()

        np_features = features.data.cpu().numpy()
        for j, item in enumerate(np_features):
            pred = predict_label(item, downsample_factor=2)
            evaluator.evaluate(pred, labels[j])

        # running_loss += current_loss.cpu().item()

    # loss = running_loss / (i + 1)
    average_evaluation_results = evaluator.get_average_results()

    return None, average_evaluation_results


def evaluate(current_experiment, currentEpoch, data_set_params):
    dataset = CostumeDataset(data_set_params.ids_path, data_set_params.data_folder_path,
                             data_set_params.labels_folder_path, img_h=224, img_w=224)
    dataloader = DataLoader(dataset)

    # Set up an experiment
    model, optimizer, optimizerScheduler, logger, epoch, lossHistory = \
        config_experiment(current_experiment, resume=True, useBest=False, currentEpoch=currentEpoch)

    if torch.cuda.is_available():
        print("Using CUDA")
        model.cuda()

    model.eval()
    average_loss, average_evaluation_results = evaluate_model(model, dataloader)

    print("Got loss: %s" % average_loss)
    print("Got evaluation results: %s" % average_evaluation_results)


def main():
    current_experiment, currentEpoch, data_set_params = validation_argument_parser()

    with torch.no_grad():
        evaluate(current_experiment, currentEpoch, data_set_params)


if __name__ == '__main__':
    main()
