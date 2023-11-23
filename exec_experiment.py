from avalanche.benchmarks.classic import SplitCIFAR100

import torch
import os
import datetime
import tqdm as tqdm

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

def exec_experiment(**kwargs):
    # Ratios of tr set used for training linear probe
    if kwargs["use_probing_tr_ratios"]:
        probing_tr_ratio_arr = [0.05, 0.1, 0.5, 1]
    else:
        probing_tr_ratio_arr = [1]


    # Set up save folders
    str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
    folder_name = f'{kwargs["model"]}_{kwargs["dataset"]}_{str_now}'
    if kwargs["iid"]:
        folder_name = 'iid_' + folder_name
    save_pth = os.path.join(kwargs["save_folder"], folder_name)
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    probing_pth_dict = {}
    for probing_tr_ratio in probing_tr_ratio_arr:  
        probing_pth = os.path.join(save_pth, f'probing_ratio{probing_tr_ratio}')
        if not os.path.exists(probing_pth):
            os.makedirs(probing_pth)
        probing_pth_dict[probing_tr_ratio] = probing_pth

    # Save general kwargs
    with open(save_pth + '/config.txt', 'a') as f:
        f.write(f'Model: {kwargs["model"]}\n')
        f.write(f'Encoder: {kwargs["encoder"]}\n')
        f.write(f'Optimizer: {kwargs["optim"]}\n')
        f.write(f'Learning Rate: {kwargs["lr"]}\n')
        f.write(f'Dataset: {kwargs["dataset"]}\n')
        f.write(f'Number of Experiences: {kwargs["num_exps"]}\n')
        f.write(f'Epochs: {kwargs["epochs"]}\n')
        f.write(f'Probing Epochs: {kwargs["probing_epochs"]}\n')
        f.write(f'Probing Use Validation Stop: {kwargs["probing_use_val_stop"]}\n')
        f.write(f'Probing Validation Ratio: {kwargs["probing_val_ratio"]}\n')
        f.write(f'Memory Size: {kwargs["mem_size"]}\n')
        f.write(f'MB Passes: {kwargs["mb_passes"]}\n')
        f.write(f'Train MB Size: {kwargs["tr_mb_size"]}\n')
        f.write(f'Replay MB Size: {kwargs["repl_mb_size"]}\n')
        f.write(f'Evaluation MB Size: {kwargs["eval_mb_size"]}\n')
        f.write(f'Use Common Transforms: {kwargs["common_transforms"]}\n')
        f.write(f'Probing Train Ratios: {probing_tr_ratio_arr}\n')
        f.write(f'IID pretraining: {kwargs["iid"]}\n')
        f.write(f'Save final model: {kwargs["save_model_final"]}\n')


    # Dataset
    first_exp_with_half_classes = False
    return_task_id = False
    shuffle = True
    probe_benchmark = SplitCIFAR100(
                kwargs["num_exps"],
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
            device = torch.device("cuda")
            print(f'There are {torch.cuda.device_count()} GPU(s) available.')
            print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")


    # Model
    if kwargs["model"] == 'no_strategy_simsiam':
        model = NoStrategySimSiam(encoder=kwargs["encoder"], optim=kwargs["optim"], train_epochs=kwargs["epochs"],
                                lr=kwargs["lr"],
                                train_mb_size=kwargs["tr_mb_size"], mb_passes=kwargs["mb_passes"],
                                dataset_name=kwargs["dataset"], save_pth=save_pth, device=device,
                                save_model=False, common_transforms=kwargs["common_transforms"])
    elif kwargs["model"] == 'replay_simsiam':
        model = ReplaySimSiam(encoder=kwargs["encoder"], optim=kwargs["optim"], mem_size=kwargs["mem_size"],
                            train_epochs=kwargs["epochs"], lr=kwargs["lr"],
                            train_mb_size=kwargs["tr_mb_size"], replay_mb_size=kwargs["repl_mb_size"],
                            mb_passes=kwargs["mb_passes"], dataset_name=kwargs["dataset"], save_pth=save_pth,
                            device=device, save_model=False, common_transforms=kwargs["common_transforms"])
        
    elif kwargs["model"] == 'replay_barlow_twins':
        model = ReplayBarlowTwins(encoder=kwargs["encoder"], optim=kwargs["optim"], train_epochs=kwargs["epochs"],
                                lr=kwargs["lr"], lambd=kwargs["lambd"],
                                mem_size=kwargs["mem_size"], train_mb_size=kwargs["tr_mb_size"],
                                mb_passes=kwargs["mb_passes"], dataset_name=kwargs["dataset"], save_pth=save_pth,
                                device=device, save_model=False, common_transforms=kwargs["common_transforms"])
    elif kwargs["model"] == 'no_strategy_barlow_twins':
        model = NoStrategyBarlowTwins(encoder=kwargs["encoder"], optim=kwargs["optim"], train_epochs=kwargs["epochs"],
                                    lr=kwargs["lr"], lambd=kwargs["lambd"],  
                                    train_mb_size=kwargs["tr_mb_size"],
                                    mb_passes=kwargs["mb_passes"], dataset_name=kwargs["dataset"], save_pth=save_pth,
                                    device=device, save_model=False, common_transforms=kwargs["common_transforms"])
    elif kwargs["model"] == 'no_strategy_byol':
        model =  NoStrategyBYOL(return_momentum_encoder=kwargs["return_momentum_encoder"], train_epochs=kwargs["epochs"],
                                lr=kwargs["lr"], byol_momentum=kwargs["byol_momentum"],
                                encoder=kwargs["encoder"], optim=kwargs["optim"], train_mb_size=kwargs["tr_mb_size"],
                                mb_passes=kwargs["mb_passes"], dataset_name=kwargs["dataset"], save_pth=save_pth,
                                device=device, save_model=False, common_transforms=kwargs["common_transforms"])
        
    elif kwargs["model"] == 'replay_byol':
        model = ReplayBYOL(return_momentum_encoder=kwargs["return_momentum_encoder"], train_epochs=kwargs["epochs"],
                                lr=kwargs["lr"], byol_momentum=kwargs["byol_momentum"],
                                encoder=kwargs["encoder"], optim=kwargs["optim"], mem_size=kwargs["mem_size"],
                                train_mb_size=kwargs["tr_mb_size"],
                                mb_passes=kwargs["mb_passes"], dataset_name=kwargs["dataset"], save_pth=save_pth,
                                device=device, save_model=False, common_transforms=kwargs["common_transforms"])
        
    else:
        # Throw exception
        raise Exception(f'Model {kwargs["model"]} not supported')

    # Self supervised training over the experiences
    for exp_idx, experience in enumerate(pretr_benchmark.train_stream):
        print(f'==== Beginning self supervised training for experience: {exp_idx} ====')
        network = model.train_experience(experience, exp_idx)

        # Do linear probing on current encoder for all experiences (past, current and future)
        for probe_exp_idx, probe_tr_experience in enumerate(probe_benchmark.train_stream):

            # Sample only a portion of the tr samples for probing
            for probing_tr_ratio in probing_tr_ratio_arr:

                probe_save_file = os.path.join(probing_pth_dict[probing_tr_ratio], f'probe_exp_{exp_idx}.csv')
                dim_features = network.get_embedding_dim() 

                # probe = LinearProbing(network.get_encoder(), dim_features=dim_features, num_classes=num_classes,
                #                     device=device, save_file=probe_save_file,
                #                     exp_idx=probe_exp_idx, tr_samples_ratio=probing_tr_ratio, num_epochs=kwargs["probing_epochs"],
                #                     use_val_stop=kwargs["probing_use_val_stop"], val_ratio=kwargs["probing_val_ratio"])
                probe = LinearProbingSklearn(network.get_encoder(), device=device, save_file=probe_save_file,
                                            exp_idx=probe_exp_idx, tr_samples_ratio=probing_tr_ratio,
                                            val_ratio=kwargs["probing_val_ratio"], mb_size=kwargs["eval_mb_size"])
                                            
                
                print(f'-- Probing on experience: {probe_exp_idx}, probe tr ratio: {probing_tr_ratio} --')

                probe.probe(probe_tr_experience, probe_benchmark.test_stream[probe_exp_idx])
                

    # Save final pretrained model
    if kwargs["save_model_final"]:
        torch.save(network.state_dict(),
                    os.path.join(save_pth, f'final_model_state.pth'))
        
    # Calculate and save final probing scores
    write_final_scores(save_pth)

    return save_pth
