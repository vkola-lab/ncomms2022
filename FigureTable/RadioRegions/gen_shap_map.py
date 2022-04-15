import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
from model_wrappers import Multask_Wrapper
from utils import read_json
from glob import glob
import numpy as np
import shutil

"""
This script is used to generate MRI model's shap map (for both COG score and ADD score) on radiologists' review cases
so that we can perform regional specific correlation between model's regionally averaged shap and radiologists' regional scores

To used the script, run the scirpt from the root dir of the repo, like below:
python FigureTable/RadioRegions/gen_shap_map.py

you can customize the FigureTable/NeuroPathSubjectABC/config.json file to adjust the experiment from which you want to load weights

the outcome shap map will be saved as numpy array in:
/data_2/sq/Radiologist/shap_mid/
"""

def gen_shap(main_config, task_config, device, tasks, cross_idx, layername):
    model_name = main_config['model_name']
    csv_dir = main_config['csv_dir']
    main_config['csv_dir'] = csv_dir + 'cross{}/'.format(cross_idx)
    main_config['model_name'] = model_name + '_cross{}'.format(cross_idx)
    for task_idx in [0, 1]:
        model = Multask_Wrapper(tasks=tasks,
                                device=device,
                                main_config=main_config,
                                task_config=task_config,
                                seed=1000)
        model.shap_mid(task_idx=task_idx,
                       path='/data_2/sq/Radiologist/shap/{}_cross{}/'.format(layername, cross_idx),
                       file='neuro_test.csv',
                       layer=layername)
        del model

def average_shap(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    cross = [folder + '_cross{}/'.format(cross_idx) for cross_idx in range(5)]
    print(cross)
    for file in glob(cross[0] + '*.npy'):
        cross_files = [file.replace('cross0', 'cross{}'.format(i)) for i in range(5)]
        print(cross_files)
        data = [np.load(f) for f in cross_files]
        data = sum(data) / 5
        np.save(folder + '/' + file.split('/')[-1], data)
    for delete_dir in cross:
        print('deleting', delete_dir)
        shutil.rmtree(delete_dir)

if __name__ == "__main__":
    layername = 'block2conv'
    # for cross_idx in range(5):
    #     gen_shap(tasks = ['COG', 'ADD'],
    #              device = 1,
    #              main_config = read_json('FigureTable/NeuroPathSubjectABC/config.json'),
    #              task_config = read_json('task_config.json'),
    #              cross_idx = cross_idx,
    #              layername = layername)
    average_shap('/data_2/sq/Radiologist/shap/{}'.format(layername))


