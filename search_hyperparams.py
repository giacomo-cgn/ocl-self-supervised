import os
import itertools
import datetime
import pandas as pd

from exec_experiment import exec_experiment
from src.utils import read_command_line_args

use_eval_on_upto_probing = True

# Parse arguments
args = read_command_line_args()

model_name = 'no_strategy_simsiam' 
# Define current searched hyperparams in lists
hyperparam_dict = {
    'lr': [0.3, 0.1, 0.03],
}
str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
folder_name = f'hypertune_{model_name}_{str_now}'
if args.iid:
     folder_name = f'hypertune_iid_{model_name}_{str_now}'
save_folder = os.path.join('./logs', folder_name)
if not os.path.exists(save_folder):
        os.makedirs(save_folder)

# Save hyperparams
with open(os.path.join(save_folder, 'hyperparams_config_results.txt'), 'w') as f:
    f.write(str(hyperparam_dict))
    f.write('\n')

# Get the keys and values from the hyperparameter dictionary
param_names = list(hyperparam_dict.keys())
param_values = list(hyperparam_dict.values())

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

     # Set args model 
     args.model = model_name

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

