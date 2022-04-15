import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from model_wrappers import Multask_Wrapper
from utils import read_json

"""
This script is used to generate MRI model's raw prediction (COG score and ADD score) on neuropath cases
so that we can perform subject level correlation between model's prediction and neurpath ABC scores

To used the script, run the scirpt from the root dir of the repo, like below:
~/multi-task$ python FigureTable/NeuroPathSubjectABC/gen_ADD_COG_scores.py

you can customize the FigureTable/NeuroPathSubjectABC/config.json file to adjust the experiment from which you want to load weights

the outcome csv files will be saved in tb_log/corresponding_exp/ as ADNI_NP_eval.csv and FHS_NP_eval.csv
"""

def crossValid(main_config, task_config, device, tasks):
    model_name = main_config['model_name']
    csv_dir = main_config['csv_dir']
    for i in range(1):
        main_config['csv_dir'] = csv_dir + 'cross{}/'.format(i)
        main_config['model_name'] = model_name + '_cross{}'.format(i)
        model = Multask_Wrapper(tasks=tasks,
                                device=device,
                                main_config=main_config,
                                task_config=task_config,
                                seed=1000)
        model.gen_score(['ADNI_NP'], join=False)
        model.gen_score(['FHS_NP'], join=False)
        model.gen_score(['np_test'], join=False)

if __name__ == "__main__":
    crossValid(tasks = ['COG', 'ADD'],
               device = 2,
               main_config = read_json('FigureTable/NeuroPathSubjectABC/config.json'),
               task_config = read_json('task_config.json'))
