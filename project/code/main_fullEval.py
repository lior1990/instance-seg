from utils.argument_parser import evaluation_argument_parser
import os
from torch import no_grad


def _runFullEval(feExp, feSubExp, clExp, clSubExp, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath):
    from fullEvaluation import run
    with no_grad():
        run(feExp, feSubExp, clExp, clSubExp, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath)


def main():
    fe_experiment, fe_sub_experiment, cl_experiment, cl_sub_experiment, feEpoch, clEpoch, dataPath, labelsPath, idsPath, outputPath, GPUs = evaluation_argument_parser()
    if GPUs:
        # should be a number or a list of comma separated numbers
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUs


    print('FE experiment:', fe_experiment)
    print('FE sub experiment:', fe_sub_experiment)
    print('CL experiment:', cl_experiment)
    print('CL sub experiment:', cl_sub_experiment)
    print('FE epoch:', feEpoch)
    print('CL epoch:', clEpoch)
    print('data folder path:', dataPath)
    print('labels folder path:', labelsPath)
    print('ids path:', idsPath)
    print('output path:', outputPath)

    fe_experiment = 'best_no_edges_no_weighted_mean'
    cl_experiment = 'best_no_edges_no_weighted_mean'
    feEpoch = '501'
    clEpoch = '5001'
    _runFullEval(fe_experiment, fe_sub_experiment, cl_experiment, cl_sub_experiment, feEpoch, clEpoch, dataPath,
                 labelsPath, idsPath, outputPath)


if __name__ == '__main__':
    main()
