import torch

import os
import datetime
import tqdm as tqdm
import numpy as np

from src.get_datasets import get_benchmark, get_iid_dataset
from src.exec_probing import exec_probing
from src.backbones import get_encoder

from src.ssl_models import BarlowTwins, SimSiam, BYOL, MoCo, SimCLR, EMP, MAE

from src.strategies import NoStrategy, Replay, ARP, AEP, APRE, LUMP, MinRed, CaSSLe
from src.standalone_strategies.scale import SCALE

from src.trainer import Trainer

from src.buffers import get_buffer

from src.utils import write_final_scores, read_command_line_args

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
        f.write(f'Probing after all experiences: {kwargs["probing_all_exp"]}\n')
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
    benchmark, image_size = get_benchmark(
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

        elif kwargs["model"] == 'moco':
            ssl_model = MoCo(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                             dim_proj=kwargs["dim_proj"],
                             moco_momentum=kwargs["moco_momentum"], moco_queue_size=kwargs["moco_queue_size"],
                             moco_temp=kwargs["moco_temp"],return_momentum_encoder=kwargs["return_momentum_encoder"],
                             queue_type=kwargs["moco_queue_type"],
                             save_pth=save_pth, device=device).to(device)
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
            


    if kwargs["iid"]:
        # IID training over the entire dataset
        print(f'==== Beginning self supervised training on iid dataset ====')
        trained_ssl_model = trainer.train_experience(iid_tr_dataset, exp_idx=0)

        exec_probing(kwargs, benchmark, trained_ssl_model.get_encoder_for_eval(), 0, probing_tr_ratio_arr, device, probing_upto_pth_dict,
                     probing_separate_pth_dict)
        
    elif kwargs["random_encoder"]:
        
        # No SSL training is done, only using the randomly initialized encoder as feature extractor
        exec_probing(kwargs, benchmark, encoder, 0, probing_tr_ratio_arr, device, probing_upto_pth_dict,
                     probing_separate_pth_dict)
    else:
        # Self supervised training over the experiences
        for exp_idx, exp_dataset in enumerate(benchmark.train_stream):
            print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
            trained_ssl_model = trainer.train_experience(exp_dataset, exp_idx)
            if kwargs["probing_all_exp"]:
                exec_probing(kwargs, benchmark, trained_ssl_model.get_encoder_for_eval(), exp_idx, probing_tr_ratio_arr, device, probing_upto_pth_dict,
                    probing_separate_pth_dict)
        if not kwargs["probing_all_exp"]:
            # Probe only at the end of training
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
