import torch

import os
import datetime
import tqdm as tqdm
import numpy as np

from torch.utils.data import random_split, Subset


from src.get_datasets import get_benchmark, get_iid_dataset
from src.exec_probing import exec_probing
from src.backbones import get_encoder

from src.ssl_models import BarlowTwins, SimSiam, BYOL, SimCLR, EMP, MAE

from src.strategies import NoStrategy, Replay, ARP, AEP, APRE, LUMP, MinRed, CaSSLe
from src.standalone_strategies.scale import SCALE

from src.trainer import Trainer

from src.buffers import get_buffer

from src.utils import write_final_scores, read_command_line_args

from src.curriculum_utils import get_gradual_subset_increase_exps, SubsetSplitterWrapper, get_classes_subset
import random

def exec_experiment(**kwargs):
    standalone_strategies = ['scale']
    buffer_free_strategies = ['no_strategy', 'aep', 'cassle']

    # Ratios of tr set used for training linear probe
    if kwargs["use_probing_tr_ratios"]:
        probing_tr_ratio_arr = [0.05, 0.1, 0.5, 1]
    else:
        probing_tr_ratio_arr = [1]


    # Set up save folders
    str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
    if kwargs["strategy"] in standalone_strategies:
        folder_name = f'{kwargs["strategy"]}_{kwargs["dataset"]}_{str_now}'
    elif kwargs['random_encoder']:
        folder_name = f'random_{kwargs["dataset"]}_{str_now}'
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
        f.write(f'---- CURRICULUM CONFIGS ----')
        f.write(f'Curriculum Order: {kwargs["curriculum_order"]}\n')
        f.write(f'Curriculum Ratio: {kwargs["curriculum_ratio"]}\n')
        f.write(f'Curriculum Subset Ratio: {kwargs["curriculum_subset"]}\n')
        f.write(f'Same Size Continual Exps:{kwargs["same_size_continual_exps"]}\n')

        

    # Set seed
    torch.manual_seed(kwargs["seed"])
    np.random.default_rng(kwargs["seed"])

    # Dataset
    benchmark, image_size = get_benchmark(
        dataset_name=kwargs["dataset"],
        dataset_root=kwargs["dataset_root"],
        num_exps=kwargs["num_exps"],
        seed=kwargs["seed"],
        val_ratio=kwargs["probing_val_ratio"],
    )

    # Curriculum
    # Curriculum Order:
    #  - "continual": train on a sequence of experiences (each with a subset of classes).
    #  - "iid": train on a single experience containing all classes. All classes in iid part
    #  - "gradual_subset": like iid but gradually increases the subset training (from  ratio 'curriculum-gradual-start' to 'curriculum-gradual-end')
    # Curriculum Ratio: ratio of training steps in each training part
    # Curriculum Subset Ratio: of the samples allocated to that training part, only use a subset of those. It is not considered 
    # for "gradual_subset" training parts

    # "--same-size-continual-exps", if true the size of the continual exps are always the same and equal to 1/num_exps of the tr length
    #   and each continual part can contain only a portion of all exps, otherwise each continual part is divided in num_exps experiences

    # IMPORTANT! Each training part is non-exclusive

    # Init the Subset splitter
    subset_splitter = SubsetSplitterWrapper(type=kwargs["subset_type"], seed=kwargs["seed"])

    tr_dataset_len = 0
    for exp_dataset in benchmark.train_stream:
        tr_dataset_len += len(exp_dataset)

    total_training_steps = int(kwargs["epochs"] * kwargs["mb_passes"] * (tr_dataset_len / kwargs["tr_mb_size"]))
    print("TOTAL TR STEPS:", total_training_steps)

    curriculum_order = kwargs["curriculum_order"].split('-')
    curriculum_ratio = [float(i) for i in kwargs["curriculum_ratio"].split('-')]
    curriculum_subset = [float(i) for i in kwargs["curriculum_subset"].split('-')]
    print("Curriculum Order:", curriculum_order)
    print("Curriculum Ratio:", curriculum_ratio)
    print("Curriculum Subset Ratio:", curriculum_subset)

    assert len(curriculum_ratio) == len(curriculum_order) == len(curriculum_subset)
    assert sum(curriculum_ratio) == 1
    for subset in curriculum_subset:
        assert subset > 0 and subset <= 1
    for curriculum_part in curriculum_order:
        assert curriculum_part in ['continual', 'iid', 'gradual_subset', 'class_subset']

    exp_list = [] # Tuple (dataset: Dataset, num_tr_steps: int, probe_after_this_tr_exp: bool)
    last_exp_idx = 0
    for i, curriculum_part in enumerate(curriculum_order):
        if curriculum_part == 'iid':
            dataset = get_iid_dataset(benchmark)
            if i > 0 and curriculum_order[i-1] == 'gradual_subset' and curriculum_subset[i] == kwargs["curriculum_gradual_end"]:
                subset = end_subset
            else:
                subset_len = int(curriculum_subset[i]*len(dataset))
                subset, _ = subset_splitter.subset(dataset, subset_len, len(dataset) - subset_len)
            tr_steps = int(curriculum_ratio[i] * total_training_steps)
            exp_list.append((subset, tr_steps, True))
            last_subset = subset

        if curriculum_part == 'continual':
            if kwargs["same_size_continual_exps"]:
                continual_steps = int(curriculum_ratio[i] * total_training_steps)
                steps_per_exp = int(total_training_steps/kwargs["num_exps"])
                
                continual_steps_allocated = 0
                for exp_idx in range(last_exp_idx, kwargs["num_exps"]):
                    last_exp_idx = exp_idx
                    if continual_steps_allocated + steps_per_exp >= continual_steps:
                        tr_steps = continual_steps - continual_steps_allocated
                        finish_continual = True
                    else:
                        tr_steps = steps_per_exp
                        finish_continual = False
                    continual_steps_allocated += tr_steps
                    exp_dataset = benchmark.train_stream[exp_idx]
                    subset_len = int(curriculum_subset[i] * len(exp_dataset))
                    subset_dataset , _ =  subset_splitter.subset(exp_dataset, subset_len, len(exp_dataset) - subset_len)
                    exp_list.append((subset_dataset, tr_steps, False))
                    if finish_continual:
                        break
            else:
                for j, exp_dataset in enumerate(benchmark.train_stream):
                    subset_len = int(curriculum_subset[i] * len(exp_dataset))
                    subset_dataset, _ = subset_splitter.subset(exp_dataset, subset_len, len(exp_dataset) - subset_len)
                    tr_steps = int((curriculum_ratio[i] * total_training_steps)/ kwargs["num_exps"])
                    exp_list.append((subset_dataset, tr_steps, False))

            exp_list[-1] = (exp_list[-1][0], exp_list[-1][1], True) # Probe after all the sequence of continual exps
        
        if curriculum_part == 'gradual_subset':
            dataset = get_iid_dataset(benchmark)
            tr_steps = int(curriculum_ratio[i] * total_training_steps)
            if i > 0 and curriculum_order[i-1] == 'iid' and curriculum_subset[i-1] == kwargs["curriculum_gradual_start"]:
                exps, end_subset = get_gradual_subset_increase_exps(dataset=dataset, total_tr_steps=tr_steps, start_subset_ratio=kwargs["curriculum_gradual_start"],
                                                    end_subset_ratio=kwargs["curriculum_gradual_end"], step_ratio=kwargs["curriculum_gradual_step"],
                                                    subset_splitter=subset_splitter, start_subset=last_subset)

            else:
                exps, end_subset = get_gradual_subset_increase_exps(dataset=dataset, total_tr_steps=tr_steps, start_subset_ratio=kwargs["curriculum_gradual_start"],
                                                    end_subset_ratio=kwargs["curriculum_gradual_end"], step_ratio=kwargs["curriculum_gradual_step"],
                                                    subset_splitter=subset_splitter)
            exp_list.extend(exps)

        if curriculum_part == 'class_subset':
            num_classes_dict = {
                'cifar10': 10,
                'cifar100': 100,
                'imagenet': 1000,
                'imagenet100': 100               
            }
            tot_classes = num_classes_dict[kwargs["dataset"]]
            tr_steps = int(curriculum_ratio[i] * total_training_steps)
            num_subset_classes = int(curriculum_subset[i] * tot_classes)
            print(f'CLASS SUBSET - {curriculum_subset[i]} of {tot_classes} classes corresponds to {num_subset_classes} classes')

            dataset = get_iid_dataset(benchmark)
            # Take num_subset_classes random classes
            subset_classes = random.sample(range(0, tot_classes), num_subset_classes)
            # Get all samples from dataset with those classes
            subset = get_classes_subset(dataset, subset_classes)
            exp_list.append((subset, tr_steps, True))
            

    for data, tr_steps, probe_after in exp_list:
        print(f"Exp len: {len(data)}, Tr steps: {tr_steps}, Probe after: {probe_after}")


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
    encoder, dim_encoder_features = get_encoder(encoder_name=kwargs["encoder"],
                                                image_size=image_size,
                                                ssl_model_name=kwargs["model"],
                                                vit_avg_pooling=kwargs["vit_avg_pooling"])
    

    if not kwargs["strategy"] in standalone_strategies:
    # SSL model
        if kwargs["model"] == 'simsiam':
            ssl_model = SimSiam(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                save_pth=save_pth).to(device)
            num_views = 2
        elif kwargs["model"] == 'byol':
            ssl_model = BYOL(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                             byol_momentum=kwargs["byol_momentum"], return_momentum_encoder=kwargs["return_momentum_encoder"],
                             save_pth=save_pth).to(device)
            num_views = 2
            
        elif kwargs["model"] == 'barlow_twins':
            ssl_model = BarlowTwins(encoder=encoder, dim_backbone_features=dim_encoder_features,
                                    dim_features=kwargs["dim_proj"],
                                    lambd=kwargs["lambd"], save_pth=save_pth).to(device)
            num_views = 2
        
        elif kwargs["model"] == 'simclr':
            ssl_model = SimCLR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"], temperature=kwargs["simclr_temp"],
                             save_pth=save_pth).to(device)
            num_views = 2

        elif kwargs["model"] == 'emp':
            ssl_model = EMP(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                            dim_proj=kwargs["dim_proj"], n_patches=kwargs["num_views"],
                            emp_tcr_param=kwargs["emp_tcr_param"], emp_tcr_eps=kwargs["emp_tcr_eps"], 
                            emp_patch_sim=kwargs["emp_patch_sim"], save_pth=save_pth).to(device)
            num_views = kwargs["num_views"]

        elif kwargs["model"] == 'mae':
            ssl_model = MAE(vit_encoder=encoder,
                            image_size=image_size, patch_size=kwargs["mae_patch_size"], emb_dim=kwargs["mae_emb_dim"],
                            decoder_layer=kwargs["mae_decoder_layer"], decoder_head=kwargs["mae_decoder_head"],
                            mask_ratio=kwargs["mae_mask_ratio"], save_pth=save_pth).to(device)

            num_views = 1
            
        else:
            raise Exception(f'Invalid model {kwargs["model"]}')
        

    # Buffer
    if not kwargs["strategy"] in buffer_free_strategies:
        if kwargs["buffer_type"] == "default":
            # Set default buffer for each strategy
            if kwargs["strategy"] in ['replay', 'apre', 'arp', 'lump']:
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


    if kwargs["aligner_dim"] <= 0:
        aligner_dim = kwargs["dim_pred"]
    else:
        aligner_dim = kwargs["aligner_dim"]
    

    if not kwargs["strategy"] in standalone_strategies:
        # Strategy
        if kwargs["strategy"] == 'no_strategy':
            strategy = NoStrategy(ssl_model=ssl_model, device=device, save_pth=save_pth)

        elif kwargs["strategy"] == 'replay':
            strategy = Replay(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
            
        elif kwargs["strategy"] == 'arp':
            strategy = ARP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim)
        
        elif kwargs["strategy"] == 'aep':
            strategy = AEP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim, momentum_ema=kwargs["momentum_ema"])
        
        elif kwargs["strategy"] == 'apre':
            strategy = APRE(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                            aligner_dim=aligner_dim, momentum_ema=kwargs["momentum_ema"])
            
        elif kwargs["strategy"] == 'scale':
            strategy = SCALE(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                            weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                            mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                            save_model=False, common_transforms=kwargs["common_transforms"],
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            dim_features=kwargs["scale_dim_features"], distill_power=kwargs["scale_distill_power"], buffer_type=kwargs["buffer_type"])
            
        elif kwargs["strategy"] == 'lump':
            strategy = LUMP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer,
                            alpha_lump=kwargs["alpha_lump"])
            
        elif kwargs["strategy"] == 'minred':
            strategy = MinRed(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
        
        elif kwargs["strategy"] == 'cassle':
            strategy = CaSSLe(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                            use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                            aligner_dim=aligner_dim)

        else:
            raise Exception(f'Strategy {kwargs["strategy"]} not supported')

        # Set up the trainer wrapper
        trainer = Trainer(ssl_model=ssl_model, strategy=strategy, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                        total_tr_steps=total_training_steps, lr_scheduler=kwargs["lr_scheduler"],
                        weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                        mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                        save_model=False, common_transforms=kwargs["common_transforms"], num_views=num_views)
        
    else:
        # Is a standalone strategy (already includes trainer and ssl model inside the strategy itself)
        trainer = SCALE(encoder=encoder, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                            weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"], train_epochs=kwargs["epochs"],
                            mb_passes=kwargs["mb_passes"], device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                            save_model=False, common_transforms=kwargs["common_transforms"],
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            dim_features=kwargs["scale_dim_features"], distill_power=kwargs["scale_distill_power"], buffer_type=kwargs["buffer_type"])
            


    
    # Self supervised training over the experiences
    for exp_idx, (exp_dataset, tr_steps, probe_after) in enumerate(exp_list):
        print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
        trained_ssl_model = trainer.train_experience(exp_dataset, exp_idx, tr_steps)

        if probe_after:
             exec_probing(kwargs, benchmark, trained_ssl_model.get_encoder_for_eval(), exp_idx, probing_tr_ratio_arr, device, probing_upto_pth_dict,
            probing_separate_pth_dict)

                
        
    # Calculate and save final probing scores
    if kwargs['probing_separate']:
        write_final_scores(folder_input_path=os.path.join(save_pth, 'probing_separate'),
                           output_file=os.path.join(save_pth, 'final_scores_separate.csv'))
    if kwargs['probing_upto']:
        write_final_scores(folder_input_path=os.path.join(save_pth, 'probing_upto'),
                           output_file=os.path.join(save_pth, 'final_scores_upto.csv'))
        
    # Save final pretrained model
    if kwargs["save_model_final"] and not kwargs["random_encoder"]:
        if kwargs['strategy'] in standalone_strategies:
            torch.save(trained_ssl_model.get_encoder_for_eval().state_dict(),
                    os.path.join(save_pth, f'final_model_state.pth'))
        else:
            torch.save(trained_ssl_model.state_dict(),
                    os.path.join(save_pth, f'final_model_state.pth'))


    return save_pth





if __name__ == '__main__':
    # Parse arguments
    args = read_command_line_args()

    exec_experiment(**args.__dict__)
