from collections import OrderedDict

from ModelWithLoss import CompleteModel
from config import embedding_dim


def load_model_from_experiment(experiment):
    model = CompleteModel(embedding_dim)

    try:
        model.load_state_dict(experiment['fe_state_dict'])
    except:
        state_dict = OrderedDict()
        prefix = 'module.'
        for key, val in experiment['fe_state_dict'].items():
            if key.startswith(prefix):
                key = key[len(prefix):]
            state_dict[key] = val
        model.load_state_dict(state_dict)

    return model
