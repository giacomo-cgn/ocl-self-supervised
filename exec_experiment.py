
import torch
from torch.utils.data import ConcatDataset
from torchvision import models

import os
import datetime
import tqdm as tqdm
import numpy as np

from src.get_datasets import get_benchmark, get_iid_dataset
from src.exec_probing import exec_probing


from src.ssl_models.barlow_twins import BarlowTwins
from src.ssl_models.simsiam import SimSiam
from src.ssl_models.byol import BYOL

from src.strategy_wrappers.no_strategy import NoStrategy
from src.strategy_wrappers.replay import Replay
from src.strategy_wrappers.align_buffer import AlignBuffer
from src.strategy_wrappers.align_ema import AlignEMA
from src.strategy_wrappers.align_ema_replay import AlignEMAReplay
from src.strategy_wrappers.lump import LUMP
from src.strategy_wrappers.minred import MinRed
from src.strategy_wrappers.cassle import CaSSLe
from src.standalone_strategies.scale import SCALE
from src.standalone_strategies.emp import EMP

from src.buffers import get_buffer

from src.utils import write_final_scores

def exec_experiment(**kwargs):
    standalone_strategies = ['scale', 'emp']
    buffer_free_strategies = ['no_strategy', 'align_ema', 'cassle', 'emp']

    # Ratios of tr set used for training linear probe
    if kwargs["use_probing_tr_ratios"]:
        probing_tr_ratio_arr = [0.05, 0.1, 0.5, 1]
    else:
        probing_tr_ratio_arr = [1]


    # Set up save folders
    str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
    if kwargs["strategy"] in standalone_strategies:
        folder_name = f'{kwargs["strategy"]}_{kwargs["dataset"]}_{str_now}'
    else:
        folder_name = f'{kwargs["strategy"]}_{kwargs["model"]}_{kwargs["dataset"]}_{str_now}'
    if kwargs["iid"]:
        folder_name = 'iid_' + folder_name
    save_pth = os.path.join(kwargs["save_folder"], folder_name)
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    
    if kwargs['probing_separate']:
        probing_separate_pth_dict = {}
        for probing_tr_ratio in probing_tr_ratio_arr:  
            probing_pth = os.path.join(save_pth, f'probing_separate/probing_ratio{probing_tr_ratio}')
            if not os.path.exists(probing_pth):
                os.makedirs(probing_pth)
            probing_separate_pth_dict[probing_tr_ratio] = probing_pth
    
    if kwargs['probing_upto']:
        probing_upto_pth_dict = {}
        for probing_tr_ratio in probing_tr_ratio_arr:  
            probing_pth = os.path.join(save_pth, f'probing_upto/probing_ratio{probing_tr_ratio}')
            if not os.path.exists(probing_pth):
                os.makedirs(probing_pth)
            probing_upto_pth_dict[probing_tr_ratio] = probing_pth


    # Save general kwargs
    with open(save_pth + '/config.txt', 'a') as f:
        f.write('\n')
        f.write(f'---- EXPERIMENT CONFIGS ----\n')
        f.write(f'Seed: {kwargs["seed"]}\n')
        f.write(f'Experiment Date: {str_now}\n')
        f.write(f'Model: {kwargs["model"]}\n')
        f.write(f'Encoder: {kwargs["encoder"]}\n')
        f.write(f'Dataset: {kwargs["dataset"]}\n')
        f.write(f'Number of Experiences: {kwargs["num_exps"]}\n')
        f.write(f'Memory Size: {kwargs["mem_size"]}\n')
        f.write(f'MB Passes: {kwargs["mb_passes"]}\n')
        f.write(f'Num Epochs: {kwargs["epochs"]}\n')
        f.write(f'Train MB Size: {kwargs["tr_mb_size"]}\n')
        f.write(f'Replay MB Size: {kwargs["repl_mb_size"]}\n')
        f.write(f'IID pretraining: {kwargs["iid"]}\n')
        f.write(f'Save final model: {kwargs["save_model_final"]}\n')
        f.write(f'-- Probing configs --\n')
        f.write(f'Probing type: {kwargs["probing_type"]}\n')
        f.write(f'Evaluation MB Size: {kwargs["eval_mb_size"]}\n')
        f.write(f'Probing on Separated exps: {kwargs["probing_separate"]}\n')
        f.write(f'Probing on joint exps Up To current: {kwargs["probing_upto"]}\n')
        f.write(f'Probing Validation Ratio: {kwargs["probing_val_ratio"]}\n')
        f.write(f'Probing Train Ratios: {probing_tr_ratio_arr}\n')
        if kwargs['probing_type'] == 'knn':
            f.write(f'KNN k: {kwargs["knn_k"]}\n')

    # Set seed
    torch.manual_seed(kwargs["seed"])
    np.random.default_rng(kwargs["seed"])

    # Dataset
    benchmark = get_benchmark(
        dataset_name=kwargs["dataset"],
        dataset_root=kwargs["dataset_root"],
        num_exps=kwargs["num_exps"],
        seed=kwargs["seed"],
        val_ratio=kwargs["probing_val_ratio"],
    )
    if kwargs["iid"]:
        iid_tr_dataset = get_iid_dataset(benchmark)   

    # Device
    if torch.cuda.is_available():       
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            if kwargs["gpu_idx"] < torch.cuda.device_count():
                device = torch.device(f"cuda:{kwargs['gpu_idx']}")
            else:
                device = torch.device("cuda")
            print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Encoder
    if kwargs["encoder"] == 'resnet18':
        encoder = models.resnet18
    elif kwargs["encoder"] == 'resnet34':
        encoder = models.resnet34
    elif kwargs["encoder"] == 'resnet50':
        encoder = models.resnet50
    else:
        raise Exception(f'Invalid encoder {kwargs["encoder"]}')
    

    if not kwargs["strategy"] in standalone_strategies:
        # Model
        if kwargs["model"] == 'simsiam':
            model = SimSiam(base_encoder=encoder, dim_proj=kwargs["dim_proj"],
                            dim_pred=kwargs["dim_pred"], save_pth=save_pth).to(device)
        elif kwargs["model"] == 'byol':
            model = BYOL(base_encoder=encoder, dim_proj=kwargs["dim_proj"],
                        dim_pred=kwargs["dim_pred"], byol_momentum=kwargs["byol_momentum"],
                        return_momentum_encoder=kwargs["return_momentum_encoder"], save_pth=save_pth).to(device)
            
        elif kwargs["model"] == 'barlow_twins':
            model = BarlowTwins(encoder=encoder, dim_features=kwargs["dim_proj"],
                                dim_pred=kwargs["dim_pred"], lambd=kwargs["lambd"], save_pth=save_pth).to(device)
            
        else:
            raise Exception(f'Invalid model {kwargs["model"]}')
        

    # Buffer
    if not kwargs["strategy"] in buffer_free_strategies:
        if kwargs["buffer_type"] == "default":
            # Set default buffer for each strategy
            if kwargs["strategy"] in ['replay', 'align_ema_replay', 'align_buffer', 'lump']:
                kwargs["buffer_type"] = "reservoir"
            elif kwargs["strategy"] == "minred":
                kwargs["buffer_type"] = "minred"
            elif kwargs["strategy"] == "scale":
                kwargs["buffer_type"] = "scale"
            
        elif kwargs["buffer_type"] == "scale" and not kwargs["strategy"] == "scale":
            raise Exception(f"Buffer type {kwargs['buffer_type']} is only compatible with strategy 'scale'")
        
        buffer = get_buffer(buffer_type=kwargs["buffer_type"], mem_size=kwargs["mem_size"],
                            alpha_ema=kwargs["features_buffer_ema"], device=device)

        # Save buffer configs
        with open(save_pth + '/config.txt', 'a') as f:
            f.write('\n')
            f.write(f'---- BUFFER CONFIGS ----\n')
            f.write(f'Buffer Type: {kwargs["buffer_type"]}\n')
            f.write(f'Buffer Size: {kwargs["mem_size"]}\n')
            if kwargs["buffer_type"] in ["minred", "reservoir", "fifo"]:
                f.write(f'Features update EMA param (MinRed): {kwargs["features_buffer_ema"]}\n')


    
    # Strategy
    if kwargs["strategy"] == 'no_strategy':
        strategy = NoStrategy(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                              weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                              mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                              save_model=False, common_transforms=kwargs["common_transforms"])
    
    elif kwargs["strategy"] == 'replay':
        strategy = Replay(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
        
    elif kwargs["strategy"] == 'align_buffer':
        strategy = AlignBuffer(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                               weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                               mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                               save_model=False, common_transforms=kwargs["common_transforms"],
                               buffer=buffer, replay_mb_size=kwargs["repl_mb_size"], omega=kwargs["omega"],
                               align_criterion=kwargs["align_criterion"], use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"])
    
    elif kwargs["strategy"] == 'align_ema':
        strategy = AlignEMA(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                            weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                            mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                            save_model=False, common_transforms=kwargs["common_transforms"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"], momentum_ema=kwargs["momentum_ema"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"])
    
    elif kwargs["strategy"] == 'align_ema_replay':
        strategy = AlignEMAReplay(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                                  weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                                  mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                                  save_model=False, common_transforms=kwargs["common_transforms"],
                                 buffer=buffer, replay_mb_size=kwargs["repl_mb_size"], omega=kwargs["omega"],
                                  align_criterion=kwargs["align_criterion"], momentum_ema=kwargs["momentum_ema"],
                                  use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"])
        
    elif kwargs["strategy"] == 'scale':
        strategy = SCALE(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                          dim_features=kwargs["scale_dim_features"], distill_power=kwargs["scale_distill_power"], buffer_type=kwargs["buffer_type"])
        
    elif kwargs["strategy"] == 'lump':
        strategy = LUMP(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          buffer=buffer,
                          alpha_lump=kwargs["alpha_lump"])
        
    elif kwargs["strategy"] == 'minred':
        strategy = MinRed(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
        
    elif kwargs["strategy"] == 'cassle':
        strategy = CaSSLe(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                            weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                            mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                            save_model=False, common_transforms=kwargs["common_transforms"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            align_after_proj=kwargs["align_after_proj"])
        
    elif kwargs["strategy"] == 'emp':
        strategy = EMP(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          n_patches=20, dim_proj=kwargs["dim_proj"]
                    )

    else:
        raise Exception(f'Strategy {kwargs["strategy"]} not supported')


    if kwargs["iid"]:
        # IID training over the entire dataset
        print(f'==== Beginning self supervised training on iid dataset ====')
        network = strategy.train_experience(iid_tr_dataset, exp_idx=0)

        exec_probing(kwargs, benchmark, network, 0, probing_tr_ratio_arr, device, probing_upto_pth_dict,
                     probing_separate_pth_dict)
    else:
        # Self supervised training over the experiences
        for exp_idx, exp_dataset in enumerate(benchmark.train_stream):
            print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
            network = strategy.train_experience(exp_dataset, exp_idx)

            exec_probing(kwargs, benchmark, network.get_encoder_for_eval(), exp_idx, probing_tr_ratio_arr, device, probing_upto_pth_dict,
                    probing_separate_pth_dict)
                
        
    # Calculate and save final probing scores
    if kwargs['probing_separate']:
        write_final_scores(folder_input_path=os.path.join(save_pth, 'probing_separate'),
                           output_file=os.path.join(save_pth, 'final_scores_separate.csv'))
    if kwargs['probing_upto']:
        write_final_scores(folder_input_path=os.path.join(save_pth, 'probing_upto'),
                           output_file=os.path.join(save_pth, 'final_scores_upto.csv'))
        
    # Save final pretrained model
    if kwargs["save_model_final"]:
        if kwargs['strategy'] in standalone_strategies:
            torch.save(network.get_encoder_for_eval().state_dict(),
                    os.path.join(save_pth, f'final_model_state.pth'))
        else:
            torch.save(network.state_dict(),
                    os.path.join(save_pth, f'final_model_state.pth'))


    return save_pth
