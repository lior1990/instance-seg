from utils.argument_parser import evaluation_argument_parser
import os
from torch import no_grad


def _runFullEval(feExp, clExp, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath):
    from fullEvaluation import run
    with no_grad():
        run(feExp, clExp, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath)


def main():
    fe_experiment, cl_experiment, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath, GPUs = evaluation_argument_parser()
    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs

    print('FE experiment:', fe_experiment)
    print('CL experiment:', cl_experiment)
    print('FE epoch:', feEpoch)
    print('CL epoch:', clEpoch)
    print('data folder path:', dataPath)
    print('labels folder path:', labelsPath)
    print('ids path:', idsPath)
    print('output path:', outputPath)
    fe_experiment = 'leafs_batch_5_cyc_lr_no_edges_dv0_5_dd10'
    cl_experiment = 'test'
    feEpoch = '1001'
    clEpoch = '3301'
    _runFullEval(fe_experiment, cl_experiment, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath)


if __name__ == '__main__':
    main()
