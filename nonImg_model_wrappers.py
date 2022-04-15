import sklearn
import os
import csv
import collections
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV
from utils import COG_thresholding, ADD_thresholding

"""
Example Usage:
    model = NonImg_Model_Wrapper(
                            tasks=['COG', 'ADD'],                            # a list of tasks to train
                            main_config=read_json('main_config.json'),       # main_config is the dict read from json
                            task_config=read_json('task_config.json'),       # task config is the dict read from json
                            seed=1000)                                       # random seed
    model.train()                                                            # train the model
    thres = model.get_optimal_thres()                                        # get optimal threshold
    model.gen_score(['test'], thres)                                         # generate csv files to future evaluation

for more details how this class is called, please see main.py

note: in the tasks argument, need to put COG before ADD since imputer will be calculated based on COG data
      and the imputer will be used to transform the ADD data 
"""

class NonImg_Model_Wrapper:
    def __init__(self, tasks, main_config, task_config, seed):

        # --------------------------------------------------------------------------------------------------------------
        # some constants
        self.seed = seed  # random seed number
        self.model_name = main_config['model_name']  # user assigned model_name, will create folder using model_name to log
        self.csv_dir = main_config['csv_dir']  # data will be loaded from the csv files specified in this directory
        self.config = task_config  # task_config contains task specific info
        self.n_tasks = len(tasks)  # number of tasks will be trained
        self.tasks = tasks  # a list of tasks names to be trained
        self.features = task_config['features'] # a list of features

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
        self.models = []  # note: self.models[i] is for the i th task
        self.init_models([task_config[t]['name'] for t in tasks])

        # --------------------------------------------------------------------------------------------------------------
        # initialize data
        self.train_data = []                      # note: self.train_data[i] contains the
        self.imputer = None
        self.load_preprocess_data()               #       features and labels for the i th task

    def train(self):
        for i, task in enumerate(self.tasks):
            X, Y = self.train_data[i].drop([task], axis=1), self.train_data[i][task]
            self.models[i].fit(X, Y)
            print(task + ' model training is done!')

    def get_optimal_thres(self, csv_name='valid'):
        self.gen_score(stages=[csv_name])
        thres = {}
        for i, task in enumerate(self.tasks):
            if task == 'COG' and self.config['COG']['type'] == 'reg':
                thres['NC'], thres['DE'] = COG_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            elif task == 'ADD':
                thres[task] = ADD_thresholding(self.tb_log_dir + csv_name + '_eval.csv')
            else:
                print("optimal for the task {} is not supported yet".format(task))
        return thres

    def gen_score(self, stages=['train', 'valid', 'test', 'OASIS'], thres={'ADD':0.5, 'NC':0.5, 'DE':1.5}):
        for stage in stages:
            data = pd.read_csv(self.csv_dir + stage + '.csv')[self.features + self.tasks + ['filename']]
            data = self.drop_cases_without_label(data, 'COG')
            COG_data = self.preprocess_pipeline(data[self.features+['COG']], 'COG') # treat it as COG data to do the preprocessing
            features = COG_data.drop(['COG'], axis=1)
            labels = data[self.tasks]
            filenames = data['filename']

            # make sure the features and labels has the same number of rows
            if len(features.index) != len(labels.index):
                raise ValueError('number of rows between features and labels have to be the same')

            predicts = []
            for i, task in enumerate(self.tasks):
                if task == 'COG':
                    predicts.append(self.models[i].predict(features))
                    print("the shape of prediction for COG task is ", predicts[-1].shape)
                if task == 'ADD':
                    predicts.append(self.models[i].predict_proba(features))
                    print("the shape of prediction for ADD task is ", predicts[-1].shape)

            content = []
            for i in range(len(features.index)):
                label = labels.iloc[i] # the feature and label are for the i th subject
                filename = filenames.iloc[i]
                case = {'filename': filename}
                for j, task in enumerate(self.tasks): # j is the task index
                    case[task] = "" if np.isnan(label[task]) else int(label[task])
                    if task == 'COG':
                        case[task+'_score'] = predicts[j][i]
                        if case[task+'_score'] < thres['NC']:
                            case[task + '_pred'] = 0
                        elif thres['NC'] <= case[task+'_score'] <= thres['DE']:
                            case[task + '_pred'] = 1
                        else:
                            case[task + '_pred'] = 2
                    elif task == 'ADD':
                        case[task + '_score'] = predicts[j][i, 1]
                        if case[task+'_score'] < thres['ADD']:
                            case[task + '_pred'] = 0
                        else:
                            case[task + '_pred'] = 1
                content.append(case)

            with open(self.tb_log_dir + stage + '_eval.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(content[0].keys()))
                writer.writeheader()
                for case in content:
                    writer.writerow(case)

    def shap(self, stage='test'):
        """
        This function will generate shap value for a specific stage
        if stage is 'test', the shap analysis will be performed on the testing section of the data
        """
        # get the data ready
        data = pd.read_csv(self.csv_dir + stage + '.csv')
        task_data = []
        for task in self.tasks:
            task_data.append(data[self.features + [task]])
        for i, task in enumerate(self.tasks):
            task_data[i] = self.preprocess_pipeline(task_data[i], task).drop([task], axis=1)

        # get the explainer ready
        self.explainer = []
        shap_values = []
        task_names = [self.config[t]['name'] for t in self.tasks]
        for i, task in enumerate(self.tasks):
            # background = shap.maskers.Independent(self.train_data[i], max_samples=100) # can we sample background from train_data?
            if task_names[i] in ['XGBoostCla', 'XGBoostReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='raw')
                    shap_values.append(explainer.shap_values(task_data[i]))
                elif 'Cla' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='predict_proba')
                    shap_values.append(explainer.shap_values(task_data[i])[1]) # index 1 means only taking ADD prob
            elif task_names[i] in ['CatBoostCla', 'CatBoostReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], model_output='raw')
                    shap_values.append(explainer.shap_values(task_data[i]))
                elif 'Cla' in task_names[i]: # use kernel explainer becaus shap only support model_output="raw"
                    explainer = shap.KernelExplainer(self.models[i].predict_proba, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=200)[1])
            elif task_names[i] in ['RandomForestCla', 'RandomForestReg', 'DecisionTreeCla', 'DecisionTreeReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='raw')
                    shap_values.append(explainer.shap_values(task_data[i]))
                elif 'Cla' in task_names[i]:
                    explainer = shap.TreeExplainer(self.models[i], task_data[i], model_output='probability')
                    shap_values.append(explainer.shap_values(task_data[i])[1])
            elif task_names[i] in ['PerceptronCla', 'PerceptronReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=500))
                elif 'Cla' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict_proba, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=500)[1]) # index 1 means only taking ADD prob
            elif task_names[i] in ['SupportVectorCla', 'SupportVectorReg', 'NearestNeighborCla', 'NearestNeighborReg']:
                if 'Reg' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=200))
                elif 'Cla' in task_names[i]:
                    explainer = shap.KernelExplainer(self.models[i].predict_proba, task_data[i])
                    shap_values.append(explainer.shap_values(task_data[i], nsamples=200)[1]) # index 1 means only taking ADD prob
            print(task + "'s shap values in shape: ", shap_values[-1].shape)
            # save the shap_values into a csv file for future use
            # rows are subjects, columns are the features
            columns = task_data[i].columns
            df = pd.DataFrame(shap_values[-1], columns=columns)
            df.to_csv(self.tb_log_dir + 'shap_'+stage+'_'+task+'.csv', index = False, header=True)
            self.shap_beeswarm_plot(shap_values[-1], task_data[i], task, stage)
        return shap_values, task_data


    ###############################################################################################################
    # below methods are internal methods and won't be called from outside of the class
    def init_models(self, task_models):
        """
        each task can have different types of models
        for example, we will use regression relevant model for the COG task
                     and classification relevant model for the ADD task
        the task_models parameter should be a python list
                     where the task_models[i] is the name of the model for the i th task
        after model initialization, models will be appended into self.models
                     where the self.models[i] is for the i th task
        """
        for name in task_models:
            # random forest model tested, function well
            if name == 'RandomForestCla':
                model = RandomForestClassifier()
            elif name == 'RandomForestReg':
                model = RandomForestRegressor()

            # xgboost model tested, function well
            elif name == 'XGBoostCla':
                model = xgb.XGBClassifier(use_label_encoder=False)
            elif name == 'XGBoostReg':
                model = xgb.XGBRegressor()

            # catboost model tested, function well
            elif name == 'CatBoostCla':
                model = CatBoostClassifier()
            elif name == 'CatBoostReg':
                model = CatBoostRegressor()

            # mlp model tested, function well
            elif name == 'PerceptronCla':
                model = MLPClassifier(max_iter=1000)
            elif name == 'PerceptronReg':
                model = MLPRegressor(max_iter=1000)

            # decision tree model tested, function well
            elif name == 'DecisionTreeCla':
                model = DecisionTreeClassifier()
            elif name == 'DecisionTreeReg':
                model = DecisionTreeRegressor()

            # support vector model tested, function well
            elif name == 'SupportVectorCla':
                model = SVC(probability=True)
            elif name == 'SupportVectorReg':
                model = SVR()

            # KNN model tested, function well
            elif name == 'NearestNeighborCla':
                model = KNeighborsClassifier()
            elif name == 'NearestNeighborReg':
                model = KNeighborsRegressor()

            self.models.append(model)

    def init_imputer(self, data):
        """
        since cases with ADD labels is only a subset of the cases with COG label
        in this function, we will initialize a single imputer
        and fit the imputer based on the COG cases from the training part
        """
        imputation_method = self.config['impute_method']
        if imputation_method == 'mean':
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif imputation_method == 'median':
            imp = SimpleImputer(missing_values=np.nan, strategy='median')
        elif imputation_method == 'most_frequent':
            imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif imputation_method == 'constant':
            imp = SimpleImputer(missing_values=np.nan, strategy='constant')
        elif imputation_method == 'KNN':
            imp = KNNImputer(n_neighbors=20)
        elif imputation_method == 'Multivariate':
            imp = IterativeImputer(max_iter=1000)
        else:
            raise NameError('method for imputation not supported')
        imp.fit(data)
        return imp

    def load_preprocess_data(self):
        data_train = pd.read_csv(self.csv_dir + 'train.csv')
        for task in self.tasks:
            self.train_data.append(data_train[self.features + [task]])
        for i, task in enumerate(self.tasks):
            self.train_data[i] = self.preprocess_pipeline(self.train_data[i], task)
            print('after preprocess pipeline, the data frame for the {} task is'.format(task))
            print(self.train_data[i])
            print('\n' * 2)

    def preprocess_pipeline(self, data, task):
        """
        Cathy, we need to remove cases with too much missing non-imaging features, please consider adding the step
        """
        # data contains features + task columns
        data = self.drop_cases_without_label(data, task)
        data = self.transform_categorical_variables(data)
        features = data.drop([task], axis=1) # drop the task columns to get all features
        features = self.imputation(features) # do imputation merely on features
        features = self.normalize(features)  # normalize features
        features[task] = data[task]          # adding the task column back
        return features                      # return the complete data

    def drop_cases_without_label(self, data, label):
        data = data.dropna(axis=0, how='any', thresh=None, subset=[label], inplace=False)
        return data.reset_index(drop=True)

    def transform_categorical_variables(self, data):
        if 'gender' in data:
            return data.replace({'male': 0, 'female': 1})
        else:
            return data
        # return pd.get_dummies(data, columns=['gender'])

    def imputation(self, data):
        columns = data.columns
        if self.imputer == None:
            self.imputer = self.init_imputer(data)
        data = self.imputer.transform(data)
        return pd.DataFrame(data, columns=columns)

    def normalize(self, data):
        df_std = data.copy()
        for column in df_std.columns:
            if data[column].std(): # normalize only when std != 0
                df_std[column] = (data[column] - data[column].mean()) / data[column].std()
        return df_std

    def shap_beeswarm_plot(self, shap_values, data, task, stage):
        from matplotlib import rc, rcParams
        rc('axes', linewidth=2)
        rc('font', weight='bold')
        fig, ax = plt.subplots(figsize=(8, 10))
        fig.text(-0.04, 0.87, 'Features', fontsize=15, fontweight='black')
        shap.summary_plot(shap_values, data)
        ax.set_xlabel('SHAP value', fontsize=15, fontweight='black')
        plt.savefig(self.tb_log_dir + '{}_shap_beeswarm_{}.png'.format(stage, task), dpi=100, bbox_inches='tight')
        plt.close()


class Fusion_Model_Wrapper(NonImg_Model_Wrapper):

    def load_preprocess_data(self):
        data_train = pd.read_csv(self.csv_dir + 'train_mri.csv')
        for task in self.tasks:
            self.train_data.append(data_train[self.features + [task]])
        for i, task in enumerate(self.tasks):
            self.train_data[i] = self.preprocess_pipeline(self.train_data[i], task)
            print('after preprocess pipeline, the data frame for the {} task is'.format(task))
            print(self.train_data[i])
            print('\n' * 2)




















