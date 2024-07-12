This library is made for executing experiments on **Continual Self-Supervised Learning**. 

It allows a wide range of experiments, thanks to its modular nature for SSL models, Continual strategies (to counter forgetting), encoders and buffers.
Moreover, it can handle both conventional (i.e. train for multiple epochs for each experience) and online continual learning setups.

## Modules

### Trainer
The Trainer module runs the training loop for each experience with the method `train_experience()`. It requires a model, strategy and the experience dataset to train. You can train for multiple epochs per experience or multiple training passes for each minibatch in the experience. It returns the SSL model trained on that experience.

### SSL model
Represent the Self Supervised Model that learns representations. Currently, SimSiam, Byol, Barlow Twins, EMP-SSL and Masked Autoencoders are implemented.
They are all subclasses of `AbstractSSLModel`. It can be specified with the command `--model`.

### Encoder
Some models may use different backbone encoders (e.g. different ResNets). Those can be specified with the command `--encoder`.
Currently implemented:
- **ResNet-18**,
- **ResNet-9**,
- **Wide ResNet-18** (2x the block features),
- **Wide ResNet-9** (2x the block features),
- **Slim ResNet-18** (~1/3 the block features),
- **Slim ResNet-9** (~1/3 the block features),
- **ViT**.


### Strategy
The strategy handles how to regularize the model to counter forgetting across experiences. All strategies are implemented as subclasses of `AbstractStrategy`.
Currently implemented:
- **No Strategy**, i.e. simple finetuning.
- **ER**, Experience Replay from buffer.
- **LUMP**, interpolate buffer with stream samples (https://arxiv.org/abs/2110.06976).
- **MinRed** training only on buffer samples, eliminates most correlated samples from buffer (to be paired with MinRed buffer) (https://arxiv.org/abs/2203.12710).
- **CaSSLe**, distillation of representations with frozen past network (https://openaccess.thecvf.com/content/CVPR2022/papers/.Fini_Self-Supervised_Models_Are_Continual_Learners_CVPR_2022_paper.pdf).
- **AEP**, distillation of representations from EMA updated network.
- **ARP**, distillation of representations from old buffer stored representations updated network.
- **APRE**, distillation of representations of replayed samples with EMA updated network.

### Buffers
Buffers store past experience samples to be replayed by strategies. Currently implemented:
- **FIFO buffer**: Stores a fixed number of samples in a FIFO queue.
- **FIFO *last* buffer**: FIFO buffer but samples only most recent samples.
- **Reservoir buffer**: Equal sampling probability for all samples in the continual stream.
- **MinRed buffer**: Removes samples with most correlated features (https://arxiv.org/abs/2203.12710).

### Probing
Evaluation on the extracted representation is done via probing. Two strategies are implemented at the moment, both uses Scikit-learn:
- Ridge Regression
- KNN

## Running experiments
A single experiment can be run with a simple command: `python main.py --option option_arg`. The full list of allowable commands is in the file `./src/utils.py`.

For running multiple experiments or running an hyperparameter search use `python run_from_config.py`. It needs a `config.json` file that specifies the configuration of each experiment, similar as the `./config.json` included in this repo.
It runs a list of experiments each with its own set of arguments; it is possible to specify common arguments for all experiments (that can be eventually overridden by each experiment). 
For each experiment desired to be run as an hyperparameter search, you need to specify  inside the experiment the additional parameter `hyperparams_search`, which is supposed to be a dict of lists of the hyperparameters to try in the experiment.





