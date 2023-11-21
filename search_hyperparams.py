import os
import itertools
import datetime
import pandas as pd

from exec_experiment import exec_experiment
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

model_name = 'no_strategy_simsiam' 
# Define current searched hyperparams in lists
hyperparam_dict = {
    'lr': [3e-3, 1e-3, 3e-4],
}
str_now = datetime.datetime.now().strftime("%m-%d_%H-%M")
folder_name = f'hypertune_{model_name}_{str_now}'
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
    results_df = pd.read_csv(os.path.join(experiment_save_folder, 'final_scores.csv'))

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


