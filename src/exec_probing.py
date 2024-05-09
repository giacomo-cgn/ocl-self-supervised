import os

from torch.utils.data import ConcatDataset

from torch.utils.data import ConcatDataset
from .probing_sklearn import ProbingSklearn


def exec_probing(kwargs, probing_benchmark, encoder, pretr_exp_idx, probing_tr_ratio_arr, device, probing_upto_pth_dict, probing_separate_pth_dict):
# Probing on all experiences up to current
    if kwargs['probing_upto'] and not (kwargs['iid'] or kwargs["random_encoder"]):
        # Generate upto current exp probing datasets
        probe_upto_dataset_tr = ConcatDataset([probing_benchmark.train_stream[i] for i in range(pretr_exp_idx+1)])
        probe_upto_dataset_test = ConcatDataset([probing_benchmark.test_stream[i] for i in range(pretr_exp_idx+1)])
        if kwargs['probing_val_ratio'] > 0:
            probe_upto_dataset_val = ConcatDataset([probing_benchmark.valid_stream[i] for i in range(pretr_exp_idx+1)])

        for probing_tr_ratio in probing_tr_ratio_arr:
            probe_save_file = os.path.join(probing_upto_pth_dict[probing_tr_ratio], f'probe_exp_{pretr_exp_idx}.csv')

            probe = ProbingSklearn(encoder, device=device, save_file=probe_save_file,
                                   exp_idx=None, tr_samples_ratio=probing_tr_ratio,
                                   mb_size=kwargs["eval_mb_size"], seed=kwargs["seed"],
                                   probing_type=kwargs["probing_type"], knn_k=kwargs["knn_k"])
                                        
            
            print(f'-- Upto Probing, probe tr ratio: {probing_tr_ratio} --')

            if kwargs['probing_val_ratio'] > 0:
                probe.probe(probe_upto_dataset_tr, probe_upto_dataset_test, probe_upto_dataset_val)
            else:
                probe.probe(probe_upto_dataset_tr, probe_upto_dataset_test)


    # Probing on separate experiences
    if kwargs['probing_separate']:
        for probe_exp_idx, probe_tr_exp_dataset in enumerate(probing_benchmark.train_stream):
            probe_test_exp_dataset = probing_benchmark.test_stream[probe_exp_idx]
            if kwargs['probing_val_ratio'] > 0:
                probe_val_exp_dataset = probing_benchmark.valid_stream[probe_exp_idx]

            # Sample only a portion of the tr samples for probing
            for probing_tr_ratio in probing_tr_ratio_arr:

                probe_save_file = os.path.join(probing_separate_pth_dict[probing_tr_ratio], f'probe_exp_{pretr_exp_idx}.csv')

                probe = ProbingSklearn(encoder, device=device, save_file=probe_save_file,
                                    exp_idx=None, tr_samples_ratio=probing_tr_ratio,
                                    mb_size=kwargs["eval_mb_size"], seed=kwargs["seed"],
                                    probing_type=kwargs["probing_type"], knn_k=kwargs["knn_k"])
                                                
                
                print(f'-- Separate Probing on experience: {probe_exp_idx}, probe tr ratio: {probing_tr_ratio} --')
                if kwargs['probing_val_ratio'] > 0:
                    probe.probe(probe_tr_exp_dataset, probe_test_exp_dataset, probe_val_exp_dataset)
                else:
                    probe.probe(probe_tr_exp_dataset, probe_test_exp_dataset)

    # If iid training, probe upto each experience
    if kwargs['probing_upto'] and (kwargs['iid'] or kwargs["random_encoder"]):
        for exp_idx, _ in enumerate(probing_benchmark.train_stream):
            # Generate upto current exp probing datasets
            probe_upto_dataset_tr = ConcatDataset([probing_benchmark.train_stream[i] for i in range(exp_idx+1)])
            probe_upto_dataset_test = ConcatDataset([probing_benchmark.test_stream[i] for i in range(exp_idx+1)])
            if kwargs['probing_val_ratio'] > 0:
                probe_upto_dataset_val = ConcatDataset([probing_benchmark.valid_stream[i] for i in range(exp_idx+1)])

            for probing_tr_ratio in probing_tr_ratio_arr:
                probe_save_file = os.path.join(probing_upto_pth_dict[probing_tr_ratio], f'probe_exp_{exp_idx}.csv')

                probe = ProbingSklearn(encoder, device=device, save_file=probe_save_file,
                                   exp_idx=None, tr_samples_ratio=probing_tr_ratio,
                                   mb_size=kwargs["eval_mb_size"], seed=kwargs["seed"],
                                   probing_type=kwargs["probing_type"], knn_k=kwargs["knn_k"])
                                            
                print(f'-- Upto Probing, probe tr ratio: {probing_tr_ratio} --')

                if kwargs['probing_val_ratio'] > 0:
                    probe.probe(probe_upto_dataset_tr, probe_upto_dataset_test, probe_upto_dataset_val)
                else:
                    probe.probe(probe_upto_dataset_tr, probe_upto_dataset_test)