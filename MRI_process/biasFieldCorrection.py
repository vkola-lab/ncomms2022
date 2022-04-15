import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from subprocess import call
import collections
import os
from glob import glob
import nibabel as nib
from pipeline import show_scan

root_dir = '/home/sq/FHS_NP_/'

print('searching for already done cases...')

good_folder = root_dir + 'good/'
good_list = glob(good_folder + '*.jpg')

tmp_folder = good_folder.replace('good', 'tmp')
scans_uniform_folder = good_folder.replace('good', 'scans_uniform')

if not os.path.exists(scans_uniform_folder):
    os.mkdir(scans_uniform_folder)

done_names = [a.split('/')[-1] for a in glob(scans_uniform_folder + '*.nii')]

for file_name in good_list:
    path = root_dir + 'scans/'
    name = file_name.split('/')[-1].replace('.jpg', '.nii')
    if name in done_names:
        continue
    call('bash biasFieldCorrection.sh '+path+' '+name+' '+scans_uniform_folder+' '+tmp_folder, shell=True)
    show_scan(file_name)
