import os
from models import _CNN_Bone, MLP, Model
from backends.ResNet import resnet18
from backends.DenseNet import DenseNet
from backends.SENet import SENet
from utils import remove_module, timeit, COG_thresholding, ADD_thresholding
from performance_eval import perform_table, ROC_PR_curves
from dataloader import TaskData, Test_Data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from scipy.special import softmax
import csv
import pandas as pd
import shap

"""
Example Usage:
    model = Multask_Wrapper(tasks=['ADD', 'COG'],                            # a list of tasks to train
                            device=1,                                        # GPU index to use
                            main_config=read_json('main_config.json'),       # main_config is the dict read from json
                            task_config=read_json('task_config.json'),       # task config is the dict read from json
                            seed=1000)                                       # random seed
    model.train()                                                            # train the model
    thres = model.get_optimal_thres()                                        # get optimal threshold
    model.gen_score(['test'], thres)                                         # generate csv files to future evaluation
    
for more details how this class is called, please see main.py
"""


class Multask_Wrapper:
    def __init__(self, tasks, device, main_config, task_config, seed, loading_data=True):

        """

        inside the __init__() method:

            initialize models
                (a) shared backbone CNN (self.backbone)
                (b) independent MLPs (self.MLPs)
                     notes: self.MLPs is a list where self.MLPs[i] is the MLP for the i th task.

            initialize loss functions
                self.losses is a list where self.losses[i] is the loss function for the i th task

            initialize optimizers
                optimizer for the backbone CNN (self.backbone_optim)
                optimizer for the MLPs (self.MLPs_optim)
                    notes: self.MLPs_optim[i] is the optimizer for the i th MLP of the i th task

            initialize dataloaders
                each task has its own dataloader, has the flexibility of controlling batch size over different tasks
                dataloaders for training part (self.train_dataloaders)
                    notes: self.train_dataloaders[i] is the pytorch dataloader for the i th task

        """

        # --------------------------------------------------------------------------------------------------------------
        # some constants
        self.device = device                                  # GPU device idx
        self.seed = seed                                      # random seed number
        self.optimal_metric = 0                               # the optimal metric value during training
        self.cur_metric = 0                                   # current metric value, if cur_metric > optimal_metric, save weights
        self.num_epochs = task_config['backbone']['epochs']   # number of epochs to train the model
        self.model_name = main_config['model_name']           # user assigned model_name, will create folder using model_name to log
        self.csv_dir = main_config['csv_dir']                 # data will be loaded from the csv files specified in this directory
        self.config = task_config                             # task_config contains task specific info
        self.n_tasks = len(tasks)                             # number of tasks will be trained
        self.tasks = tasks                                    # a list of tasks names to be trained

        # --------------------------------------------------------------------------------------------------------------
        # folders preparation to save checkpoints of model weights *.pth
        self.checkpoint_dir = './checkpoint_dir/{}/'.format(self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        # folders preparation to save tensorboard and other logs
        self.tb_log_dir = './tb_log/{}/'.format(self.model_name)
        if not os.path.exists(self.tb_log_dir):
            os.mkdir(self.tb_log_dir)

        # --------------------------------------------------------------------------------------------------------------
        # initialize models
        self.backbone_model = task_config['backbone']['model']
        if self.backbone_model == "CNN":
            self.backbone = _CNN_Bone(self.config['backbone']).to(self.device)
        elif self.backbone_model == "ResNet18":
            self.backbone = resnet18().to(self.device)
        elif self.backbone_model == "DenseNet":
            self.backbone = DenseNet().to(self.device)
        elif self.backbone_model == "SENet":
            self.backbone = SENet(self.config['backbone']).to(self.device)
        print(self.backbone)

        for task in tasks:
            if task not in self.config:                     # if the task was not specified in the self.config
                self.config[task] = self.config['default']  # use default task setting
        self.MLPs = [MLP(self.backbone.size, self.config[t]).to(self.device) for t in tasks]
        print(self.MLPs)

        # --------------------------------------------------------------------------------------------------------------
        # loss and optimizer
        self.losses = self.get_losses(tasks)
        self.backbone_optim = optim.Adam(self.backbone.parameters(), lr=self.config['backbone']['lr'], betas=(0.5, 0.999))
        self.MLPs_optim = [optim.Adam(self.MLPs[i].parameters(), lr=self.config[tasks[i]]['lr']) for i in range(self.n_tasks)]

        # --------------------------------------------------------------------------------------------------------------
        # prepare train dataloaders
        if loading_data:
            self.train_dataloaders = []
            self.prepare_dataloader()


    def train(self):
        self.writer = SummaryWriter(self.tb_log_dir)
        for self.epoch in range(self.num_epochs):
            self.train_an_epoch()
            self.cur_metric = self.valid_an_epoch()
            if self.needToSave():
                self.saveWeights()
            self.adjust_learning_rate()


    def get_optimal_thres(self, csv_name='valid'):
        """
        since the default csv_name is valid, thus we will get optimal threshold using validation set
        if the csv_name == 'train', then the training set will be used to calculate the optimal threshold

        in a binary classification task: a single threshold is needed to binarize the continuous probability score

        in the COG task, we chose to use regression task to train the model
            label value for NC=0, MCI=1, DE=2
            during inference, model predict a single scalar number
                a threshold between NC and MCI will be used to separate NC from MCI + DE
                a threshold between MCI and DE will be used to separate DE from NC + MCI

        in the end of the function, all the thresholds will be returned as a dictionary, where key is the name of the task
        and content is the corresponding threshold value.

        for example if thres = {'ADD':0.56, 'NC':0.67, 'DE':1.35}
            0.56 separates nADD and ADD for the "ADD" task
            0.67 separates NC from MCI+DE among the "COG" task
            1.45 separates DE from NC+MCI among the "COG" task
        """
        thres = {}
        for i, task in enumerate(self.tasks):
            if task == 'COG' and self.config['COG']['type'] == 'reg':
                thres['NC'], thres['DE'] = COG_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            elif task == 'ADD':
                thres[task] = ADD_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            else:
                print("optimal for the task {} is not supported yet".format(task))
        return thres


    def gen_score(self, stages=['train', 'valid', 'test', 'OASIS'], thres={'ADD':0.5, 'NC':0.5, 'DE':1.5}, load_weight=True, join=True):
        """
        gen_score will write model's predictions into a csv file for future model evaluation

        evaluations to be performed:

            1. ROC curves
                a. NC vs not NC (NC: normal cognition)
                b. DE vs not DE (DE: dementia)
                c. ADD vs nADD (ADD: Alzheimer's disease dementia) (nADD: non-AD dementia)

            2. confusion matrices
                a. 3 by 3 (NC, MCI, DE)              -- for the COG task
                b. 2 by 2 (ADD, nADD)                -- for the ADD task
                c. 4 by 4 (NC, MCI, ADD, nADD)       -- for the combination

            3. performance metrics are based on confusion matrices
                a. NC (sensitivity; specificity, etc)
                b. MCI (sensitivity; specificity, etc)
                c. DE (sensitivity; specificity, etc)
                d. ADD (sensitivity; specificity, etc)
                e. nADD (sensitivity; specificity, etc)
                f. ADD | DE (sensitivity; specificity, etc)     identify ADD given the condition that the case must be DE
                g. nADD | DE (sensitivity; specificity, etc)    identify nADD given the condition that the case must be DE

        To satisfy above requirements,
        a csv file with the following format will be generated:
        ---------------------------------------------------------------------------
        | ID | filename | COG_score | ADD_score | COG_pred | ADD_pred | COG | ADD |
        ---------------------------------------------------------------------------

        for details about how to analyse the csv file, please see performance_eval.py
        """
        self.set_train_status(False)
        if load_weight:
            self.loadWeights()
        with torch.no_grad():
            for stage in stages:
                content = []
                dataloader = DataLoader(Test_Data(self.csv_dir + stage + '.csv'), batch_size=1, shuffle=False)
                for mri, name in dataloader:
                    case = {'filename': name[0].split('/')[-1]}
                    middle = self.backbone(mri.to(self.device))
                    for i, task in enumerate(self.tasks):
                        output = self.MLPs[i](middle).data.cpu().squeeze().numpy()
                        if task == 'COG':
                            case[task + '_score'] = output
                            if output < thres['NC']:
                                case[task + '_pred'] = 0
                            elif thres['NC'] <= output <= thres['DE']:
                                case[task + '_pred'] = 1
                            else:
                                case[task + '_pred'] = 2
                        else:
                            case[task + '_score'] = softmax(output)[1]
                            case[task + '_pred'] = 0 if softmax(output)[1] < thres['ADD'] else 1
                    content.append(case)
                with open(self.tb_log_dir + stage + '_eval.csv', 'w') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
                    writer.writeheader()
                    for case in content:
                        writer.writerow(case)
                if join:
                    # join the original table with this currently generated to add labels
                    extra_content = pd.read_csv(self.csv_dir + stage + '.csv')[["COG", "ADD", "filename"]]
                    content = pd.read_csv(self.tb_log_dir + stage + '_eval.csv')
                    result = pd.merge(content, extra_content, how="left", on=["filename"])
                    result.to_csv(self.tb_log_dir + stage + '_eval.csv', index=False)

    def gen_embd(self, stages=['test'], layer='early', load_weight=True, join=True, fold=0):
        """
        gen_embd will save model's embeddings as numpy array
	the input of MLP model will be considered as embedding
        """
        self.set_train_status(False)
        if load_weight:
            self.loadWeights()
        with torch.no_grad():
            for stage in stages:
                dataloader = DataLoader(Test_Data(self.csv_dir + stage + '.csv'), batch_size=1, shuffle=False)
                for mri, name in dataloader:
                    case = {'filename': name[0].split('/')[-1]}
                    middle = self.backbone(mri.to(self.device))
                    if layer == 'late':
                        middle0 = self.MLPs[0](middle, get_intermediate_score=False).data.cpu().squeeze().numpy()
                        middle1 = self.MLPs[1](middle, get_intermediate_score=False).data.cpu().squeeze().numpy()
                        middle = np.concatenate(([middle0], softmax(middle1)[1:]))
                        print(middle.shape, case['filename'])
                        np.save("./t_SNE/late_embeddings/"+case['filename'], middle)
                    elif layer == 'mid':
                        middle0 = self.MLPs[0](middle, get_intermediate_score=True).data.cpu().squeeze().numpy()
                        middle1 = self.MLPs[1](middle, get_intermediate_score=True).data.cpu().squeeze().numpy()
                        middle = np.concatenate((middle0, middle1))
                        print(middle.shape, case['filename'])
                        np.save("./t_SNE/mid_embeddings/"+case['filename'], middle)
                    elif layer == 'early':
                        middle = middle.data.cpu().squeeze().numpy().ravel()
                        print(middle.shape)
                        np.save("./t_SNE/early_embeddings/"+case['filename'], middle)
                    else:
                        print('bad')
            
    def shap(self, task_idx=1):
        path = '/data_2/sq/'
        print("started the shap analysis for the CNN model respect to MRI ... ")
        if not os.path.exists(path + 'shap/'): # create the folder for storing shap heatmaps
            os.mkdir(path + 'shap/')
        task = self.tasks[task_idx]
        print("explaining the {} task".format(task))
        self.set_train_status(False)
        self.loadWeights()
        model = Model(self.backbone, self.MLPs[task_idx], task).to(self.device)
        # get some background cases to initialize the explainer
        background = []
        data = Test_Data(csv_file = self.csv_dir + task + '_shap_background.csv')
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for mri, name in dataloader:
            background.append(mri)
        background = torch.cat(background, 0).to(self.device)
        print("to initialize shap explainer, background shape is ", background.shape)
        e = shap.DeepExplainer(model, background)
        del background
        print("deep explainer successfully initialized")
        # explain the data and save as numpy array
        data = Test_Data(csv_file = self.csv_dir + 'test.csv')
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for mri, name in dataloader:
            mri = mri.to(self.device)
            name = name[0].split('/')[-1]
            shap_value = e.shap_values(mri)
            if task == 'ADD':
                print('shap_value\'s shape for the AD probability is', shap_value[1].shape)
                print('saved numpy array size is ', shap_value[1].squeeze().shape)
                np.save(path + 'shap/shap_{}_{}'.format(task, name), shap_value[1].squeeze())
            elif task == 'COG':
                # print('shap_value\'s shape for the COG score is', shap_value.shape)
                # print('after summing over channel dimension, shape is', shap_value.squeeze().sum(0).shape)
                np.save(path + 'shap_mid/shap_{}_{}'.format(task, name), shap_value.squeeze().sum(0))

    def shap_mid(self, task_idx=1, path='/data_2/sq/', file='test.csv', background_idx=1, layer='block2conv'):
        print("started the shap analysis for the CNN model respect to the {} layer's input ... ".format(layer))
        if not os.path.exists(path): # create the folder for storing shap heatmaps
            os.mkdir(path)
        task = self.tasks[task_idx]
        print("explaining the {} task".format(task))
        self.set_train_status(False)
        self.loadWeights()
        model = Model(self.backbone, self.MLPs[task_idx], task).to(self.device)
        # get some background cases to initialize the explainer
        background = []
        data = Test_Data(csv_file=self.csv_dir + task + '_shap_background{}.csv'.format(background_idx))
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for mri, _ in dataloader:
            background.append(mri)
        background = torch.cat(background, 0).to(self.device)
        print("to initialize shap explainer, background shape is ", background.shape)
        if layer == 'block2conv':
            e = shap.DeepExplainer((model, model.backbone.block2.conv), background)
        elif layer == 'block2pooling':
            e = shap.DeepExplainer((model, model.backbone.block2.pooling), background)
        elif layer == 'block2BN':
            e = shap.DeepExplainer((model, model.backbone.block2.BN), background)
        else:
            raise ValueError('layer not added yet, add another elif by yourself in shap_mid method')
        del background
        print("deep explainer successfully initialized")
        # explain the data and save as numpy array
        data = Test_Data(csv_file=self.csv_dir + file)
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for mri, name in dataloader:
            mri = mri.to(self.device)
            name = name[0].split('/')[-1]
            shap_value = e.shap_values(mri)
            if task == 'ADD':
                print('shap_value\'s shape for the AD probability is', shap_value[1].shape)
                print('after summing over channel dimension, shape is', shap_value[1].squeeze().sum(0).shape)
                np.save(path + 'shap_{}_{}'.format(task, name), shap_value[1].squeeze().sum(0))
            elif task == 'COG':
                print('shap_value\'s shape for the COG score is', shap_value.shape)
                print('after summing over channel dimension, shape is', shap_value.squeeze().sum(0).shape)
                np.save(path + 'shap_{}_{}'.format(task, name), shap_value.squeeze().sum(0))

    ###############################################################################################################
    # below methods are internal methods and won't be called from outside of the class
    def get_losses(self, tasks):
        losses = []
        for task in tasks:
            if self.config[task]['type'] == 'cla':
                losses.append(nn.CrossEntropyLoss().to(self.device))
            elif self.config[task]['type'] == 'reg':
                losses.append(nn.MSELoss().to(self.device))
        return losses

    def cast_labels(self, label, task):
        if self.config[task]['type'] == 'reg':
            return label.float().view(-1, 1)
        elif self.config[task]['type'] == 'cla':
            return label.long()
        return label

    def prepare_dataloader(self, ratio={}):
        patch_ = (None, None) if self.backbone_model != "FCN" else ("random", "random")
        for task in self.tasks:
            batch_size = self.config[task]['batch_size']
            train_data = TaskData(task, task_config=self.config[task], csv_dir=self.csv_dir, stage='train', seed=self.seed, patch=patch_[0])
            sample_weight = train_data.get_sample_weights(ratio)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, len(sample_weight))
            self.train_dataloaders.append(DataLoader(train_data, batch_size=batch_size, sampler=sampler, drop_last=True, num_workers=1))
        self.iter_per_epoch = min(len(dataloader) for dataloader in self.train_dataloaders)

    @timeit
    def train_an_epoch(self):
        self.set_train_status(True)
        iter_train_dataloaders = {i: iter(self.train_dataloaders[i]) for i in range(self.n_tasks)}
        for _ in range(self.iter_per_epoch):
            self.zero_grad_all()
            for i in range(self.n_tasks):
                inputs, labels, names = next(iter_train_dataloaders[i])
                preds, loss, labels = self.forward_task(i, inputs, labels)
                loss.backward()
            self.update_all_optim()

    @timeit
    def valid_an_epoch(self, metric='AUC'):
        self.gen_score(['valid'], load_weight=False)  # using default thres value to generate score for validation set
        if metric == 'MCC':
            thres = self.get_optimal_thres('valid')       # get optimal thres for validation
            self.gen_score(['valid'], thres, load_weight=False) # apply the optimal thres on validation to generate score and pred
            metric = perform_table([self.tb_log_dir+'valid_eval.csv'])
        elif metric == 'AUC':
            metric = ROC_PR_curves([self.tb_log_dir+'valid_eval.csv'], 'valid')
        return metric

    def forward_task(self, i, inputs, labels):
        labels = self.cast_labels(labels, self.tasks[i])
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        preds = self.MLPs[i](self.backbone(inputs))
        loss = self.losses[i](preds, labels) * float(self.config[self.tasks[i]]['factor'])
        return preds, loss, labels

    def set_train_status(self, mode):
        for model in self.MLPs:
            model.train(mode)
        self.backbone.train(mode)

    def zero_grad_all(self):
        for model in self.MLPs:
            model.zero_grad()
        self.backbone.zero_grad()

    def update_all_optim(self):
        self.backbone_optim.step()
        for optim in self.MLPs_optim:
            optim.step()

    def saveWeights(self, clean_previous=True):
        if clean_previous:
            files = glob(self.checkpoint_dir+'*.pth')
            for f in files:
                os.remove(f)
        torch.save(self.backbone.state_dict(), '{}backbone_{}.pth'.format(self.checkpoint_dir, self.epoch))
        for task, model in zip(self.tasks, self.MLPs):
            torch.save(model.state_dict(), '{}{}_{}.pth'.format(self.checkpoint_dir, task, self.epoch))

    def loadWeights(self):
        self.load_backbone_Weights()
        self.load_MLP_weights()

    def load_backbone_Weights(self):
        target_file = list(glob(self.checkpoint_dir + 'backbone*.pth'))[0]
        print('loading ', target_file)
        weights = torch.load(target_file, map_location=lambda storage, loc: storage.cuda(self.device))
        try:
            self.backbone.load_state_dict(weights)
        except:
            self.backbone.load_state_dict(remove_module(weights))

    def load_MLP_weights(self):
        for i, task in enumerate(self.tasks):
            target_file = list(glob(self.checkpoint_dir + task + '*.pth'))[0]
            print('loading ', target_file)
            try:
                self.MLPs[i].load_state_dict(torch.load(target_file))
            except:
                self.MLPs[i].load_state_dict(remove_module(torch.load(target_file)))

    def needToSave(self):
        if self.cur_metric > self.optimal_metric:
            self.optimal_metric = self.cur_metric
            return True
        return False

    def adjust_learning_rate(self):
        if self.epoch in [round(self.num_epochs * 0.333), round(self.num_epochs * 0.666)]:
            for param_group in self.backbone_optim.param_groups:
                param_group['lr'] *= 0.2
            for optim in self.MLPs_optim:
                for param_group in optim.param_groups:
                    param_group['lr'] *= 0.2
            print('Adjust_learning_rate')


if __name__ == "__main__":
    model = resnet18()
    print(model)




