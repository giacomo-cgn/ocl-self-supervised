from avalanche.benchmarks.classic import SplitCIFAR100

import torch
import os
import datetime
import tqdm as tqdm

from exec_experiment import exec_experiment

from src.replay_simsiam import ReplaySimSiam
from src.replay_barlow_twins import ReplayBarlowTwins
from src.no_strategy_simsiam import NoStrategySimSiam
from src.no_strategy_barlow_twins import NoStrategyBarlowTwins
from src.no_strategy_byol import NoStrategyBYOL
from src.replay_byol import ReplayBYOL

from src.transforms import get_dataset_transforms
from src.probing import LinearProbing
from src.probing_sklearn import LinearProbingSklearn
from src.utilities import write_final_scores

import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='no_strategy_simsiam')
parser.add_argument('--encoder', type=str, default='resnet18')
parser.add_argument('--optim', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--num-exps', type=int, default=20)
parser.add_argument('--save-folder', type=str, default='./logs')
parser.add_argument('--probing-epochs', type=int, default=50)
parser.add_argument('--probing-use-val-stop', type=bool, default=True)
parser.add_argument('--probing-val-ratio', type=float, default=0.1)
parser.add_argument('--mem-size', type=int, default=2000)
parser.add_argument('--mb-passes', type=int, default=3)
parser.add_argument('--tr-mb-size', type=int, default=32)
parser.add_argument('--repl-mb-size', type=int, default=32)
parser.add_argument('--eval-mb-size', type=int, default=1024)
parser.add_argument('--common-transforms', type=bool, default=True)
parser.add_argument('--use-probing-tr-ratios', type=bool, default=False)
parser.add_argument('-iid', '--iid', type=bool, default=False)
parser.add_argument('--save-model-final', type=bool, default=True)

# Models specific params
parser.add_argument('--lambd', type=float, default=5e-3)
parser.add_argument('--byol-momentum', type=float, default=0.9)
parser.add_argument('--return-momentum-encoder', type=bool, default=True)


args = parser.parse_args()

exec_experiment(**args.__dict__)