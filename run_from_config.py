import json
import copy
import os

from src.utils import read_command_line_args
from search_hyperparams import search_hyperparams


# Read args from command line
original_args = read_command_line_args()

# Read config.json
with open('config.json') as f:
    config = json.load(f)

    for k, v in config["common_params"].items():
        original_args.__setattr__(k, v)
        # Add also variant with param name with "-" substituted with "_" and vice versa
        original_args.__setattr__(k.replace("_", "-"), v)
        original_args.__setattr__(k.replace("-", "_"), v)

    # Params for identifying this experiment in logs
    log_dir = os.path.join('logs', f'mbtr{original_args.tr_mb_size}_mbrep{original_args.repl_mb_size}_k{original_args.mb_passes}')


    for experiment in config["experiments"]:
        args = copy.deepcopy(original_args)

        # Apply experiment specific params
        for k, v in experiment.items():
            if k is not "hyperparams_search":
                args.__setattr__ (k, v)
                # Add also variant with param name with "-" substituted with "_" and vice versa
                args.__setattr__(k.replace("_", "-"), v)
                args.__setattr__(k.replace("-", "_"), v)
        
        # Run hyperparam search
        search_hyperparams(args, hyperparams_dict=experiment["hyperparams_search"], parent_log_folder=log_dir)