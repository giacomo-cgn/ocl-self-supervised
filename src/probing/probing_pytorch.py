import copy
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn


class Appr(Learning_Appr):
    """
    Based on the implementation for pytorch-lighting:
    github.com:zlapp/pytorch-lightning-bolts.git
    """

    def __init__(
            self,
            model,
            device,
            nepochs=100,
            lr=0.05,
            lr_min=1e-4,
            lr_factor=3,
            lr_patience=5,
            clipgrad=10000,
            momentum=0,
            wd=1e-6,
            multi_softmax=False,
            wu_nepochs=0,
            wu_lr_factor=1,
            fix_bn=False,
            eval_on_train=False,
            logger=None,
            exemplars_dataset=None,
            # approach params
            warmup_epochs=0,
            lr_warmup_epochs=10,
            hidden_mlp: int = 2048,
            feat_dim: int = 128,
            # maxpool1 = False,
            # first_conv = False,
            # input_height = 32,
            temperature=0.5,
            gaussian_blur=False,
            jitter_strength=0.4,
            optim_name='sgd',
            lars_wrapper=True,
            exclude_bn_bias=False,
            start_lr: float = 0.,
            final_lr: float = 0.,
            classifier_nepochs=20,
            incremental_lr_factor=0.1,
            eval_nepochs=100,
            head_classifier_lr=5e-3,
            head_classifier_min_lr=1e-6,
            head_classifier_lr_patience=3,
            head_classifier_hidden_mlp=2048,
            init_after_each_task=True,
            kd_method='ft',
            p2_hid_dim=512,
            pred_like_p2=False,
            joint=False,
            diff_lr=False,
            change_lr_scheduler=False,
            lambdapRet=1.0,
            lambdaExp=1.0,
            task1_nepochs=500,
            wandblog=False,
            loadTask1=False,
            lamb=0.01,
            projectorArc='8192_8192_8192',
            batch_size=512,
            lambd=0.0051,
            dataset2='cifar100',
            pathModelT1='',
            port='11',
            ret_nepochs=500,
            reInit=False,
            loadExpert=False,
            pathExperts="",
            reInitExpert=False,
            expertArch="ResNet9",
            saveExpert='',
            extImageNet=False,
            expProjSize=512,
            trans_nepochs=0,
            adaptSche=0,
            linearProj=0,
            getResults = False,
            projWarm = 50,
            lrExpF = 1,
            norm = False,
            loadm2 = False,
            sslModel = "BT"

    ):
        super(Appr, self).__init__(
            model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd, multi_softmax,
            wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger, exemplars_dataset
        )

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.hidden_mlp = hidden_mlp
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.gaussian_blur = gaussian_blur
        self.jitter_strength = jitter_strength
        self.optim_name = optim_name
        self.lars_wrapper = lars_wrapper
        self.exclude_bn_bias = exclude_bn_bias
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.lr_warmup_epochs = lr_warmup_epochs
        self.classifier_nepochs = classifier_nepochs
        self.incremental_lr_factor = incremental_lr_factor
        self.eval_nepochs = eval_nepochs
        self.head_classifier_lr = head_classifier_lr
        self.head_classifier_min_lr = head_classifier_min_lr
        self.head_classifier_lr_patience = head_classifier_lr_patience
        self.head_classifier_hidden_mlp = head_classifier_hidden_mlp
        self.init_after_each_task = init_after_each_task
        self.kd_method = kd_method
        self.p2_hid_dim = p2_hid_dim
        self.pred_like_p2 = pred_like_p2
        self.diff_lr = diff_lr
        self.change_lr_scheduler = change_lr_scheduler
        self.lambdapRet = lambdapRet
        self.lambdaExp = lambdaExp
        self.task1_nepochs = task1_nepochs
        self.loadTask1 = loadTask1
        self.projectorArc = projectorArc
        self.batch_size = batch_size
        self.lambd = lambd
        self.dataset2 = dataset2
        self.pathModelT1 = pathModelT1
        self.port = port
        self.ret_nepochs = ret_nepochs
        self.reInit = reInit
        self.loadExpert = loadExpert
        self.pathExperts = pathExperts
        self.reInitExpert = reInitExpert
        self.expertArch = expertArch
        self.saveExpert = saveExpert
        self.extImageNet = extImageNet
        self.expProjSize = expProjSize
        self.trans_nepochs = trans_nepochs
        self.adaptSche = adaptSche
        self.linearProj = linearProj
        self.getResults = getResults
        self.projWarm = projWarm
        self.lrExpF = lrExpF
        self.norm = norm
        self.loadm2 = loadm2
        self.sslModel = sslModel

        # Logs
        self.wandblog = wandblog

        # internal vars
        self._step = 0
        self._encoder_emb_dim = 512
        self._task_classifiers = []
        self._task_classifiers_update_step = -1
        self._task_classifiers_update_step = -1
        self._current_task_dataset = None
        self._current_task_classes_num = None
        self._online_train_eval = None
        self._initialized_net = None
        self._tbwriter: SummaryWriter = self.logger.tbwriter

        # Lightly
        self.gpus = [torch.cuda.current_device()]
        self.distributed_backend = 'ddp' if len(self.gpus) > 1 else None

        # LwF lambda
        self.lamb = np.ones((10, 1)) * lamb
        self.expertAccu = []

        # save embeddings
        self.embeddingAvai = np.zeros((10, 1))
        self.trainX = {}
        self.trainXexp = {}
        self.trainY = {}
        self.valX = {}
        self.valXexp = {}
        self.valY = {}

        # Joint
        self.joint = joint
        if self.joint:
            print('Joint training!')
            self.trn_datasets = []
            self.val_datasets = []

        # Wandb for log purposes
        import pandas as pd
        # Load it into a dataframe
        d = {'nepochs': str(nepochs),
             'head_classifier_lr': str(self.head_classifier_lr),
             'task1_nepochs': str(self.task1_nepochs),
             'kd_method': self.kd_method,
             'lambdapRet': str(self.lambdapRet),
             'lambdaExp': str(self.lambdaExp),
             'classifier_nepochs': str(self.classifier_nepochs),
             'dataset': self.dataset2,
             'projectorArc': self.projectorArc,
             'reInit': self.reInit,
             'reInitExpert': self.reInitExpert
             }
        parameters = pd.DataFrame(data=d, index=[0])


# Extract embeddings only once per task
def get_embeddings(self, encoder, t, trn_loader, val_loader):
    # Get backbone
    # modelT = deepcopy(self.modelFB.backbone).to(self.device)
    # modelT = deepcopy(self.modelFB.currentExpert).to(self.device)
    modelT = copy.deepcopy(encoder).to(self.device)
    modelTexpert = copy.deepcopy(encoder).to(self.device)
    for param in modelT.parameters():
        param.requires_grad = False
    for param in modelTexpert.parameters():
        param.requires_grad = False
    modelT.eval();
    modelTexpert.eval()

    # Create tensors to store embeddings
    batchFloorT = (len(trn_loader.dataset) // trn_loader.batch_size) * trn_loader.batch_size if \
        (len(trn_loader.dataset) // trn_loader.batch_size) * trn_loader.batch_size != 0 else len(trn_loader.dataset)
    batchFloorV = len(val_loader.dataset)

    trainX = torch.zeros((batchFloorT, self._encoder_emb_dim), dtype=torch.float).to(self.device)
    trainY = torch.zeros(batchFloorT, dtype=torch.long).to(self.device)
    valX = torch.zeros((batchFloorV, self._encoder_emb_dim), dtype=torch.float).to(self.device)
    valY = torch.zeros(batchFloorV, dtype=torch.long).to(self.device)

    trainXexp = torch.zeros((batchFloorT, self._encoder_emb_dim), dtype=torch.float).to(self.device)
    valXexp = torch.zeros((batchFloorV, self._encoder_emb_dim), dtype=torch.float).to(self.device)

    with override_dataset_transform(trn_loader.dataset, self.val_transforms) as _ds_train, \
            override_dataset_transform(val_loader.dataset, self.test_transforms) as _ds_val:
        _train_loader = DataLoader(
            _ds_train,
            batch_size=trn_loader.batch_size,
            shuffle=False,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
            drop_last=True
        )
        _val_loader = DataLoader(
            _ds_val,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory
        )

        contBatch = 0
        # import pdb;
        # pdb.set_trace()

        for img_1, y in _train_loader:

            _xexp = modelTexpert(img_1.to(self.device)).flatten(start_dim=1)
            # _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
            _x = _xexp
            _x = _x.detach();
            _xexp = _xexp.detach()
            y = torch.LongTensor((y - self.model.task_offset[t]).long().cpu()).to(self.device)
            # y = torch.LongTensor((y-25*t).long().cpu()).to(self.device)

            trainX[contBatch:contBatch + trn_loader.batch_size, :] = _x
            trainY[contBatch:contBatch + trn_loader.batch_size] = y
            trainXexp[contBatch:contBatch + trn_loader.batch_size, :] = _xexp
            contBatch += trn_loader.batch_size

        contBatch = 0
        for img_1, y in _val_loader:
            # _x = modelT(img_1.to(self.device)).flatten(start_dim=1)
            _xexp = modelTexpert(img_1.to(self.device)).flatten(start_dim=1)
            _x = _xexp
            _x = _x.detach();
            _xexp = _xexp.detach()
            y = torch.LongTensor((y - self.model.task_offset[t]).long().cpu()).to(self.device)
            # y = torch.LongTensor((y -25*t).long().cpu()).to(self.device)
            valX[contBatch:contBatch + _val_loader.batch_size, :] = _x
            valY[contBatch:contBatch + _val_loader.batch_size] = y
            valXexp[contBatch:contBatch + _val_loader.batch_size, :] = _xexp
            contBatch += _val_loader.batch_size

    return torch.nn.functional.normalize(trainX), \
        trainY, \
        torch.nn.functional.normalize(valX), \
        valY, \
        torch.nn.functional.normalize(trainXexp), \
        torch.nn.functional.normalize(valXexp)


class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.out_features = n_classes  # for *head* compability
        if n_hidden is None or n_hidden == 0:
            # use linear classifier
            self.model = nn.Sequential(nn.Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.model(x)
        return logits




def train_classifier(self, encoder, t, trn_loader, val_loader, name='classifier'):
    # Extract embeddings
    trainX, trainY, valX, valY, trainXexp, valXexp = get_embeddings(t, encoder, trn_loader, val_loader)
    self.trainX[str(t)] = trainX
    self.trainY[str(t)] = trainY
    self.valX[str(t)] = valX
    self.valY[str(t)] = valY
    self.trainXexp[str(t)] = trainXexp
    self.valXexp[str(t)] = valXexp

    # prepare classifier
    _class_lbl = sorted(np.unique(trn_loader.dataset.labels).tolist())
    _num_classes = len(_class_lbl)
    # MLP
    # _task_classifier = SSLEvaluator(self._encoder_emb_dim, _num_classes, self.hidden_mlp, 0.0)
    # Linear
    _task_classifier = SSLEvaluator(self._encoder_emb_dim, _num_classes, 0, 0.0)
    _task_classifierexp = SSLEvaluator(self._encoder_emb_dim, _num_classes, 0, 0.0)

    _task_classifier.to(self.device)
    _task_classifierexp.to(self.device)
    lr = self.head_classifier_lr
    _task_classifier_optimizer = torch.optim.Adam(_task_classifier.parameters(), lr=lr)
    # _task_classifier_optimizer = torch.optim.SGD(_task_classifier.parameters(), lr=lr)
    _task_classifier_optimizerexp = torch.optim.Adam(_task_classifierexp.parameters(), lr=lr)

    # train on train dataset after learning representation of task
    classifier_train_step = 0
    val_step = 0
    best_val_loss = 1e10
    best_val_acc = 0.0
    patience = self.lr_patience
    _task_classifier.train()
    _task_classifierexp.train()
    best_model = None

    for e in range(self.classifier_nepochs):

        # train
        train_loss = 0.0
        train_lossexp = 0.0
        train_samples = 0.0
        index = 0

        while index + trn_loader.batch_size <= self.trainX[str(t)].shape[0]:
            _x = self.trainX[str(t)][index:index + trn_loader.batch_size, :]
            y = self.trainY[str(t)][index:index + trn_loader.batch_size]

            _x = _x.detach()
            # forward pass
            mlp_preds = _task_classifier(_x.to(self.device))
            mlp_loss = F.cross_entropy(mlp_preds, y)
            # update finetune weights
            mlp_loss.backward()
            _task_classifier_optimizer.step()
            _task_classifier_optimizer.zero_grad()
            train_loss += mlp_loss.item()
            train_samples += len(y)


            classifier_train_step += 1
            index += trn_loader.batch_size

        train_loss = train_loss / train_samples

        # eval on validation
        _task_classifier.eval()
        _task_classifierexp.eval()
        val_loss = 0.0
        acc_correct = 0
        acc_all = 0
        with torch.no_grad():
            singelite = False if self.valX[str(t)].shape[0] > val_loader.batch_size else True
            index = 0
            while index + val_loader.batch_size < self.valX[str(t)].shape[0] or singelite:
                _x = self.valX[str(t)][index:index + val_loader.batch_size, :]
                _xexp = self.valXexp[str(t)][index:index + val_loader.batch_size, :]
                y = self.valY[str(t)][index:index + val_loader.batch_size]
                _x = _x.detach();
                _xexp = _x.detach()
                # forward pass
                mlp_preds = _task_classifier(_x.to(self.device))
                mlp_loss = F.cross_entropy(mlp_preds, y)
                val_loss += mlp_loss.item()
                n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                n_all = y.size()[0]
                _val_acc = n_corr / n_all
                # print(f"{self.name} online acc: {train_acc}")
                self.logger.log_scalar(task=t, iter=val_step, name=name + '-val-acc', value=_val_acc, group="val")
                acc_correct += n_corr
                acc_all += n_all
                self.logger.log_scalar(
                    task=t, iter=val_step, name=f"{name}-val-loss", value=mlp_loss.item(), group="val"
                )
                val_step += 1
                index += val_loader.batch_size
                singelite = False

        # main validation loss
        val_loss = val_loss / acc_all
        val_acc = acc_correct / acc_all

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f'| Epoch {e} | Train loss: {train_loss:.6f} | Valid loss: {val_loss:.6f} acc: {100 * val_acc:.2f} |',
            end=''
        )

        # Adapt lr
        if val_loss < best_val_loss or best_model is None:
            best_val_loss = val_loss
            best_model = copy.deepcopy(_task_classifier.model.state_dict())
            patience = self.lr_patience
            print('*', end='', flush=True)
        else:
            # print('', end='', flush=True)
            patience -= 1
            if patience <= 0:
                lr /= self.lr_factor
                print(' lr={:.1e}'.format(lr), end='')
                if lr < self.lr_min:
                    print(' NO MORE PATIENCE')
                    break
                patience = self.lr_patience
                _task_classifier_optimizer.param_groups[0]['lr'] = lr
                _task_classifier.model.load_state_dict(best_model)
        self.logger.log_scalar(task=t, iter=e + 1, name=f"{name}-patience", value=patience, group="train")
        self.logger.log_scalar(task=t, iter=e + 1, name=f"{name}-lr", value=lr, group="train")
        print()

    _task_classifier.model.load_state_dict(best_model)
    _task_classifier.eval()
    print(f'{name} - Best ACC: {100 * best_val_acc:.1f}')

    return _task_classifier
    