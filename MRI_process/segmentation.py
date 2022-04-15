from subprocess import call
from glob import glob

path = '/data_2/sq/Radiologist/mri'
root_folder = '/data_2/sq/Radiologist'

processed = glob('/data_2/sq/Radiologist/seg/*.nii')

for file in glob(path + '/*.nii'):
    name = file.split('/')[-1]
    if file.replace('/mri/', '/seg/') in processed:
        print('processed and continue')
        continue
    call('bash segmentation.sh ' + path + ' ' + name + ' ' + root_folder, shell=True)

# import csv
# import nibabel as nib
# import numpy as np
# from tqdm import tqdm
#
# with open('/home/sq/ssl-brain/lookupcsv/CrossValid/exp4/test.csv', 'r') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in tqdm(reader):
#         data = np.load(row['path'] + row['filename'])
#         data = nib.Nifti1Image(data, affine=np.eye(4))
#         nib.save(data, '/data_2/sq/NACC_ALL_seg/mri4/' + row['filename'].replace('.npy', '.nii'))
