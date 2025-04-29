import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from einops import rearrange

from .transforms import get_transforms


class FeatureDeviationAnalyzer():
    def __init__(self,
                 when_features_deviation: str = '50_end',
                 dataset_name: str = 'cifar100',
                 transforms_type: str = 'common',
                 num_views: int = 50,
                 num_current_samples: int = 500,
                 mb_size: int=10,
                 device: str ='cpu',
                 save_pth: str = None,
                 ):
        
        split_by_underscore = lambda s: s.split('_')

        
        self.when_features_deviation = split_by_underscore(when_features_deviation)
        self.mb_size = mb_size
        self.num_current_samples = num_current_samples
        self.device = device

        if transforms_type == 'common':
            self.transforms = get_transforms(dataset=dataset_name, n_crops=num_views, online_transforms=True)
        else:
            raise Exception(f'Transforms type {self.transforms_type} not supported')
        
        if save_pth is not None:
            feat_analyze_pth = os.path.join(save_pth, 'feature_deviation')
            os.makedirs(feat_analyze_pth, exist_ok=True)
            self.e_save_pth = os.path.join(feat_analyze_pth, 'e_features.csv')
            self.z_save_pth = os.path.join(feat_analyze_pth, 'z_features.csv')
            with open(self.e_save_pth, 'a') as f:
                f.write('exp_idx,tr_step,avg_e_curr_std,avg_e_buff_std\n')
            with open(self.z_save_pth, 'a') as f:
                f.write('exp_idx,tr_step,avg_z_curr_std,avg_z_buff_std\n') 

            with open(save_pth + '/config.txt', 'a') as f:
                f.write('\n')
                f.write('---- ANALYZE FEATURES DEVIATION CONFIG ----\n')
                f.write(f'when (training steps) to analyze: {self.when_features_deviation}\n')
                f.write(f'num analyzed views: {num_views}\n')
                f.write(f'num current samples: {num_current_samples}\n')
                f.write(f'mb size analysis: {self.mb_size}\n')

                
        else:
            self.e_save_pth = None
            self.z_save_pth = None


    


    def analyze_features_deviation(self, encoder, buffer_data, current_data, exp_idx, tr_step, projector=None):

        print(f'>>> Analyzing features deviation at step {tr_step}')

        buffer_dataloader = self.prepare_buffer_data(buffer_data)
        current_dataloader = self.prepare_current_data(current_data)

        encoder.eval()
        if projector is not None:
            projector.eval()

        with torch.no_grad():

            e_std_curr_list, z_std_curr_list = [], [],
            e_std_buff_list, z_std_buff_list = [], []

            # Extract features from current data
            print('>>> Extracting features from current data, len', len(current_dataloader))
            for _, current_mbatch in enumerate(tqdm(current_dataloader)):
                current_mbatch = current_mbatch.to(self.device)
                current_mbatch_views = self.transforms(current_mbatch)

                # Convert from list of minibatch views to list of per sample views
                V = torch.stack(current_mbatch_views, dim=0) # first stack current_mbatch_views → (v, b, d1, d2, …)
                per_sample_views_list = list(rearrange(V, 'v b ... -> b v ...').unbind(dim=0)) # then rearrange to (b, v, d1, d2, …) and unbind

                for x_views in per_sample_views_list:
                    e_views = encoder(x_views)
                    # compute mean and std of e_views
                    e_std_curr_list.append(torch.mean(torch.std(e_views, dim=0)))

                    if projector is not None:
                        z_views = projector(e_views)
                        # compute mean and std of z_views
                        z_std_curr_list.append(torch.mean(torch.std(z_views, dim=0)))

            # Extract features from buffer data
            print('>>> Extracting features from buffer data, len', len(buffer_dataloader))
            for _, buffer_mbatch in enumerate(tqdm(buffer_dataloader)):
                buffer_mbatch = buffer_mbatch.to(self.device)
                buffer_mbatch_views = self.transforms(buffer_mbatch)

                # Convert from list of minibatch views to list of per sample views
                V = torch.stack(buffer_mbatch_views, dim=0) # first stack buffer_mbatch_views → (v, b, d1, d2, …)
                per_sample_views_list = list(rearrange(V, 'v b ... -> b v ...').unbind(dim=0)) # then rearrange to (b, v, d1, d2, …) and unbind

                for x_views in per_sample_views_list:
                    e_views = encoder(x_views)
                    # compute mean and std of e_views
                    e_std_buff_list.append(torch.mean(torch.std(e_views, dim=0)))

                    if projector is not None:
                        z_views = projector(e_views)
                        # compute mean and std of z_views
                        z_std_buff_list.append(torch.mean(torch.std(z_views, dim=0)))         


            # Calculate avg of statistics over all samples and save
            if self.e_save_pth is not None:
                avg_e_std_curr = torch.mean(torch.stack(e_std_curr_list), dim=0)
                avg_e_std_buff = torch.mean(torch.stack(e_std_buff_list), dim=0)
                with open(self.e_save_pth, 'a') as f:
                    f.write(f'{exp_idx},{tr_step},{avg_e_std_curr.item()},{avg_e_std_buff.item()}\n')

            if projector is not None and self.z_save_pth is not None:
                avg_z_std_curr = torch.mean(torch.stack(z_std_curr_list), dim=0)
                avg_z_std_buff = torch.mean(torch.stack(z_std_buff_list), dim=0)
                with open(self.z_save_pth, 'a') as f:
                    f.write(f'{exp_idx},{tr_step},{avg_z_std_curr.item()},{avg_z_std_buff.item()}\n')
            
            
        encoder.train()
        if projector is not None:
            projector.train()


    def prepare_buffer_data(self, buffer_data):
        # If buffer_data is a list, convert it to a tensor, stack along a new dim
        if isinstance(buffer_data, list):
            buffer_data = torch.stack(buffer_data, dim=0)
        buffer_dataloader = DataLoader(buffer_data, batch_size=self.mb_size, shuffle=False)
            
        return buffer_dataloader
    
    def prepare_current_data(self, current_data):
        # Select self.num_current_samples samples from current_data Dataset
        subset_current_data = Subset(current_data, np.random.choice(len(current_data), self.num_current_samples, replace=False))
        current_dataloader = DataLoader(subset_current_data, batch_size=self.mb_size, shuffle=False)

        return current_dataloader
    
    def get_when_to_analyze(self, tr_step, total_steps):
        """
        Determines whether feature deviation analysis should be performed at the current training step.
        Args:
            tr_step (int): Current training step.
            total_steps (int): Total number of training steps.

        Returns:
            bool: True if analysis should be performed at this step, False otherwise.
                 Returns True if:
                 - It's the final step (tr_step == total_steps - 1) and 'end' is in when_features_deviation
                 - The current step number (as string) is in when_features_deviation
        """
        if tr_step == (total_steps-1):
            if 'end' in self.when_features_deviation:
                return True
        if  str(tr_step) in self.when_features_deviation:
            return True
        return False
        
            








                



