import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
print(parentdir)
sys.path.append(parentdir)
from performance_eval import *

######################################################### fusion model
# model_name = '_XGBoost'
# print(model_name + ' Fusion performance')
# prefix = parentdir + '/tb_log/{}_Fusion_cross'.format(model_name)
# csv_files = [prefix + str(i) + '/test_mri_eval_thres.csv' for i in range(5)]
# perform_table(csv_files, '{}0/performance_test'.format(prefix))
# csv_files = [prefix + str(i) + '/OASIS_mri_eval_thres.csv' for i in range(5)]
# perform_table(csv_files, '{}0/performance_OASIS'.format(prefix))

combinations = ["demo", "demo+np", "demo+func", "demo+his",
                "demo+np+func", "demo+np+his", "demo+np+his+func",
                "MRI+demo", "MRI+demo+np", "MRI+demo+func", "MRI+demo+his",
                "MRI+demo+np+func", "MRI+demo+np+his", "MRI+demo+np+his+func"]
for model_name in combinations:
    prefix = parentdir + '/tb_log/{}_cross'.format(model_name)
    csv_files = [prefix + str(i) + '/test_mri_eval_thres.csv' for i in range(5)]
    perform_table(csv_files, '{}0/performance_test'.format(prefix))
    csv_files = [prefix + str(i) + '/OASIS_mri_eval_thres.csv' for i in range(5)]
    perform_table(csv_files, '{}0/performance_OASIS'.format(prefix))

## MRI model
# model_name = 'CNN_baseline_new'
# print(model_name + ' MRI performance')
# prefix = parentdir + '/tb_log/{}_cross'.format(model_name)
# csv_files = [prefix + str(i) + '/test_eval.csv' for i in range(5)]
# perform_table(csv_files, '{}_MRI_NACC_test'.format(model_name))
# csv_files = [prefix + str(i) + '/OASIS_eval.csv' for i in range(5)]
# perform_table(csv_files, '{}_MRI_OASIS'.format(model_name))
# csv_files = [prefix + str(i) + '/exter_test_eval.csv' for i in range(5)]
# perform_table(csv_files, '{}_MRI_exter'.format(model_name))

## non imaging model
# model_name = '_CatBoost'
# print(model_name + ' NonImg performance')
# prefix = parentdir + '/tb_log/{}_NonImg_cross'.format(model_name)
# csv_files = [prefix + str(i) + '/test_eval.csv' for i in range(5)]
# perform_table(csv_files, '{}0/performance_test'.format(prefix))
# csv_files = [prefix + str(i) + '/OASIS_eval.csv' for i in range(5)]
# perform_table(csv_files, '{}0/performance_OASIS'.format(prefix))

# Fusion model on 100 cases for neurologists
# model_name = 'XGBoost_Fusion'
# print(model_name + ' performance')
# # csv_files = [parentdir + '/tb_log/CatBoost_special1_Fusion_cross{}/neuro_test_mri_eval.csv'.format(i) for i in range(5)]
# csv_files = ['tmp/test_{}.csv'.format(i) for i in range(5)]
# perform_table(csv_files, '{}'.format(model_name + '_test'))
# csv_files = ['tmp/oasis_{}.csv'.format(i) for i in range(5)]
# perform_table(csv_files, '{}'.format(model_name + '_oasis'))

# csv_files = ['tmp/exter_{}.csv'.format(i) for i in range(5)]
# perform_table(csv_files, '{}'.format(model_name + '_exter'))
# neurologists
# model_name = 'Neurologists_100'
# print(model_name + ' performance')
# csv_files = [parentdir + '/FigureTable/NeuroROC/neurologists/n{}_new.csv'.format(i) for i in range(1, 18)]
# perform_table(csv_files, '{}'.format(model_name))
