import os
import itertools
import datetime
import pandas as pd

from exec_experiment import exec_experiment

def search_hyperparams(args, hyperparams_dict=None, use_eval_on_upto_probing=True, parent_log_folder='./logs', experiment_name=''):

     standalone_strategies = ['scale']

     if args.probing_val_ratio == 0.0:
          print('WARNING! - probing_val_ratio is 0, cannot execute hyperparams search. Exiting this experiment...')
          return
          

     # model_name = 'no_strategy_simsiam' 
     if hyperparams_dict is None:
          # Define current searched hyperparams in lists
          hyperparams_dict = {
          'lr': [0.1, 0.01, 0.001, 0.0001],
          # 'byol-momentum': [0.99, 0.999],
          }
          print('WARNING! - Hyperparams of the experiments not found, using default values:')
          print(hyperparams_dict)
     
     str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")

     if args.strategy in standalone_strategies:
          folder_name = f'hypertune_{experiment_name}_{args.strategy}_{str_now}'
     else:     
          folder_name = f'hypertune_{experiment_name}_{args.strategy}_{args.model}_{str_now}'
     
     if args.iid:
          folder_name = f'hypertune_iid_{experiment_name}_{args.strategy}_{args.model}_{str_now}'
     save_folder = os.path.join(parent_log_folder, folder_name)
     if not os.path.exists(save_folder):
          os.makedirs(save_folder)

     # Save hyperparams
     with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'w') as f:
          f.write(str(hyperparams_dict))
          f.write('\n')

     # Get the keys and values from the hyperparameter dictionary
     param_names = list(hyperparams_dict.keys())
     param_values = list(hyperparams_dict.values())

     # Generate all combinations of hyperparameters
     param_combinations = list(itertools.product(*param_values))

     best_val_acc = 0
     # Iterate through each combination and execute the train function
     for combination in param_combinations:
          param_dict = dict(zip(param_names, combination))
          print('<<<<<<<<<<<<<<< Executing experiment with:', param_dict, '>>>>>>>>>>>>>>>>>')

          # Update args with hyperparams
          for k, v in param_dict.items():
               args.__setattr__(k, v)
               # Add also variant with param name with "-" substituted with "_" and vice versa
               args.__setattr__(k.replace("_", "-"), v)
               args.__setattr__(k.replace("-", "_"), v)

          # Set args model 
          # args.model = model_name

          # Set args save_folder
          args.save_folder = save_folder

          # Execute experiment
          experiment_save_folder = exec_experiment(**args.__dict__)

          # Recover results from experiment
          if use_eval_on_upto_probing:
               results_df = pd.read_csv(os.path.join(experiment_save_folder, 'final_scores_upto.csv'))
          else:
               results_df = pd.read_csv(os.path.join(experiment_save_folder, 'final_scores_separate.csv'))

          # Only row with probe_ratio = 1
          results_df = results_df[results_df['probe_ratio'] == 1]

          val_acc = results_df['avg_val_acc'].values[0]
          test_acc = results_df['avg_test_acc'].values[0]

          with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'a') as f:
               f.write(f"{param_dict}, Val Acc: {val_acc}, Test Acc: {test_acc} \n")

          if val_acc > best_val_acc:
               best_val_acc = val_acc
               best_test_acc = test_acc
               best_combination = param_dict
     

     print(f"Best hyperparameter combination found: {best_combination}")
     # Save to file best combination of hyperparams, test and val accuracies
     with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'a') as f:
          f.write(f"\nBest hyperparameter combination: {best_combination}\n")
          f.write(f"Best Val Acc: {best_val_acc}\n")
          f.write(f"Best Test Acc: {best_test_acc}\n")
          f.write(f'\nTr MB size: {args.tr_mb_size}\n')
          f.write(f'MB passes: {args.mb_passes}\n')


