from avalanche.benchmarks.classic import SplitCIFAR100

import torch
from torch.utils.data import ConcatDataset
from torchvision import models

import os
import datetime
import tqdm as tqdm

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

from src.transforms import get_dataset_transforms
from src.probing_sklearn import ProbingSklearn
from src.utils import write_final_scores

def exec_experiment(**kwargs):
    standalone_strategies = ['scale']

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

    # Dataset
    first_exp_with_half_classes = False
    return_task_id = False
    shuffle = True
    probe_benchmark = SplitCIFAR100(
                kwargs["num_exps"],
                seed=42, # Fixed seed for reproducibility
                first_exp_with_half_classes=first_exp_with_half_classes,
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(kwargs["dataset"]),
                eval_transform=get_dataset_transforms(kwargs["dataset"]),
            )
    if kwargs["iid"]:
        # If pretraining iid, create benchmark with only 1 experience
        pretr_benchmark  = SplitCIFAR100(
                1,
                seed=42, # Fixed seed for reproducibility
                first_exp_with_half_classes=first_exp_with_half_classes,
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=get_dataset_transforms(kwargs["dataset"]),
                eval_transform=get_dataset_transforms(kwargs["dataset"]),
        )
    else:
        # Use same benchmark for pretraining and probing
        pretr_benchmark = probe_benchmark
            

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
                          mem_size=kwargs["mem_size"], replay_mb_size=kwargs["repl_mb_size"])
        
    elif kwargs["strategy"] == 'align_buffer':
        strategy = AlignBuffer(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                               weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                               mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                               save_model=False, common_transforms=kwargs["common_transforms"],
                               mem_size=kwargs["mem_size"], replay_mb_size=kwargs["repl_mb_size"], omega=kwargs["omega"],
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
                                  mem_size=kwargs["mem_size"], replay_mb_size=kwargs["repl_mb_size"], omega=kwargs["omega"],
                                  align_criterion=kwargs["align_criterion"], momentum_ema=kwargs["momentum_ema"],
                                  use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"])
        
    elif kwargs["strategy"] == 'scale':
        strategy = SCALE(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          mem_size=kwargs["mem_size"], replay_mb_size=kwargs["repl_mb_size"],
                          dim_features=kwargs["scale_dim_features"], distill_power=kwargs["scale_distill_power"], use_scale_buffer=kwargs["use_scale_buffer"])
        
    elif kwargs["strategy"] == 'lump':
        strategy = LUMP(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          mem_size=kwargs["mem_size"],
                          alpha_lump=kwargs["alpha_lump"])
        
    elif kwargs["strategy"] == 'minred':
        strategy = MinRed(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                          weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                          mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                          save_model=False, common_transforms=kwargs["common_transforms"],
                          mem_size=kwargs["mem_size"], replay_mb_size=kwargs["repl_mb_size"],
                          minred_buffer_ema=kwargs["minred_buffer_ema"])
        
    elif kwargs["strategy"] == 'cassle':
        strategy = CaSSLe(model=model, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                            weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                            mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                            save_model=False, common_transforms=kwargs["common_transforms"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            align_after_proj=kwargs["align_after_proj"])

    else:
        raise Exception(f'Strategy {kwargs["strategy"]} not supported')

    # Self supervised training over the experiences
    for exp_idx, experience in enumerate(pretr_benchmark.train_stream):
        print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
        network = strategy.train_experience(experience, exp_idx)

        # Probing on all experiences up to current
        if kwargs['probing_upto'] and not kwargs['iid']:
            # Generate upto current exp probing datasets
            probe_upto_dataset_tr = ConcatDataset([probe_benchmark.train_stream[i].dataset for i in range(exp_idx+1)])
            probe_upto_dataset_test = ConcatDataset([probe_benchmark.test_stream[i].dataset for i in range(exp_idx+1)])

            for probing_tr_ratio in probing_tr_ratio_arr:
                probe_save_file = os.path.join(probing_upto_pth_dict[probing_tr_ratio], f'probe_exp_{exp_idx}.csv')

                probe = ProbingSklearn(network.get_encoder_for_eval(), device=device, save_file=probe_save_file,
                                            exp_idx=None, tr_samples_ratio=probing_tr_ratio,
                                            val_ratio=kwargs["probing_val_ratio"], mb_size=kwargs["eval_mb_size"],
                                            probing_type=kwargs["probing_type"], knn_k=kwargs["knn_k"])
                                            
                
                print(f'-- Upto Probing, probe tr ratio: {probing_tr_ratio} --')

                probe.probe(probe_upto_dataset_tr, probe_upto_dataset_test)


        # Probing on separate experiences
        if kwargs['probing_separate']:
            for probe_exp_idx, probe_tr_experience in enumerate(probe_benchmark.train_stream):
                probe_test_experience = probe_benchmark.test_stream[probe_exp_idx]

                # Sample only a portion of the tr samples for probing
                for probing_tr_ratio in probing_tr_ratio_arr:

                    probe_save_file = os.path.join(probing_separate_pth_dict[probing_tr_ratio], f'probe_exp_{exp_idx}.csv')

                    # dim_features = network.get_embedding_dim() 
                    # probe = LinearProbing(network.get_encoder_for_eval(), dim_features=dim_features, num_classes=100,
                    #                     device=device, save_file=probe_save_file,
                    #                     exp_idx=probe_exp_idx, tr_samples_ratio=probing_tr_ratio, num_epochs=kwargs["probing_epochs"],
                    #                     use_val_stop=kwargs["probing_use_val_stop"], val_ratio=kwargs["probing_val_ratio"])
                    probe = ProbingSklearn(network.get_encoder_for_eval(), device=device, save_file=probe_save_file,
                                                exp_idx=probe_exp_idx, tr_samples_ratio=probing_tr_ratio,
                                                val_ratio=kwargs["probing_val_ratio"], mb_size=kwargs["eval_mb_size"],
                                                probing_type=kwargs["probing_type"], knn_k=kwargs["knn_k"])
                                                
                    
                    print(f'-- Separate Probing on experience: {probe_exp_idx}, probe tr ratio: {probing_tr_ratio} --')

                    probe.probe(probe_tr_experience.dataset, probe_test_experience.dataset)
    
        # If iid training, probe upto each experience
        if kwargs['probing_upto'] and kwargs['iid']:
            for exp_idx, _ in enumerate(probe_benchmark.train_stream):
                # Generate upto current exp probing datasets
                probe_upto_dataset_tr = ConcatDataset([probe_benchmark.train_stream[i].dataset for i in range(exp_idx+1)])
                probe_upto_dataset_test = ConcatDataset([probe_benchmark.test_stream[i].dataset for i in range(exp_idx+1)])

                for probing_tr_ratio in probing_tr_ratio_arr:
                    probe_save_file = os.path.join(probing_upto_pth_dict[probing_tr_ratio], f'probe_exp_{exp_idx}.csv')

                    probe = ProbingSklearn(network.get_encoder_for_eval(), device=device, save_file=probe_save_file,
                                                exp_idx=None, tr_samples_ratio=probing_tr_ratio,
                                                val_ratio=kwargs["probing_val_ratio"], mb_size=kwargs["eval_mb_size"],
                                                probing_type=kwargs["probing_type"], knn_k=kwargs["knn_k"])
                                                
                    
                    print(f'-- Upto Probing, probe tr ratio: {probing_tr_ratio} --')

                    probe.probe(probe_upto_dataset_tr, probe_upto_dataset_test)
                
        
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
