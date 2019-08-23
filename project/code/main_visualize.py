import os
import torch
from utils.argument_parser import validation_argument_parser

def _visualize(current_experiment,currentEpoch,dataPath,labelsPath,idsPath):
    # this must be imported after setting CUDA_VISIBLE_DEVICES environment variable, otherwise it won't work
    from visualize import run
    with torch.no_grad():
        run(current_experiment,currentEpoch,dataPath,labelsPath,idsPath)



def main():
    current_experiment, currentEpoch, dataPath, labelsPath, idsPath, GPUs = validation_argument_parser()
    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    with torch.no_grad():
        _visualize(current_experiment,currentEpoch, dataPath, labelsPath, idsPath)


if __name__ == '__main__':
    main()
