import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from glob import glob
from subprocess import call
import os   
from glob import glob
import json


def read_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def getID(filename):
    if '.nii' in filename:
        return filename.strip('.nii')
    elif '.jpg' in filename:
        return filename.strip('.jpg')

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def show_scan(path, filename, output_folder, mode_flag):
    image_name = filename.replace('nii', 'jpg')
    data = nib.load(path + '/' + filename).get_data()
    plt.subplot(1, 3, 1)
    plt.imshow(data[95, :, :])
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(data[:, 100, :])
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(data[:, :, 100])
    plt.colorbar()
    plt.savefig(output_folder+'/'+mode_flag+'/'+image_name)
    plt.close()

def find_cases(folder, extention):
    hashmap = {}
    for root, dirs, files in os.walk(folder, topdown=False):
        for file in files:
            if file.endswith(extention):
                hashmap[getID(file)] = os.path.join(root, file)
    return hashmap

def main(raw_folder, root_folder, mode_flag, step_7_f, step_7_g):
    """
    :param mode_flag: 'images', 'top', 'bottom', 'bone', ...
    :param check_processed:
    :param id_range:
    :param step_7_command:
    :return:
    """
    raw_dict = find_cases(raw_folder, '.nii')

    # create folder
    scan_folder = root_folder + '/scans'
    create_folder(scan_folder)
    create_folder(root_folder + '/images')
    create_folder(root_folder + '/tmp')
    create_folder(root_folder + '/'+ mode_flag)

    if mode_flag == 'images': # if mode_flag is "images", run until all processed cases are saved in the images folder
        print('Total number of cases: ', len(raw_dict))
        processed_dict = find_cases(scan_folder, '.nii')
        print('cases processed: ', len(processed_dict))
        file_dict = {key:raw_dict[key] for key in raw_dict if key not in processed_dict} # needs to be processed
        if len(raw_dict) == len(processed_dict):
            print('all cases have been processed')
            return
    else: # mode_flag is either 'top', 'bottom', 'bone', or folders named by other mode_flags
        # process cases only in the mode folder
        mode_folder = scan_folder.replace('scans', mode_flag)
        image_dict = find_cases(mode_folder, '.jpg')
        good_dict = find_cases(root_folder + '/good', '.jpg')
        print('There are {} good cases already processed', len(good_dict))
        file_dict = {key : raw_dict[key] for key in image_dict if key.replace('.nii', '.jpg') not in good_dict}
    print('start processing in ' + mode_flag + ' with ' + str(len(file_dict)) + ' cases.')
    for key in file_dict:
        path_name = file_dict[key]
        path = os.path.dirname(path_name)
        name = path_name.split('/')[-1]      
        call('bash pipeline.sh ' + path + ' ' + name + ' ' + root_folder + ' ' + step_7_f + ' ' + step_7_g, shell=True)
        show_scan(scan_folder, name, root_folder, mode_flag)
        show_scan(scan_folder, name, root_folder, 'images')

if __name__ == "__main__":
    config = read_json('pipeline_config.json')
    print(config)
    main(config['rawPath'], config['rootPath'], config['mode'], config['step7F'], config['step7G'])

"""
workflow:
1. run the main.py with default pipeline_config.json to process all cases. processed nifti will be saved in scans/; 
   images will be ploted in images/ for the inspection of processing quality. 
2. review the cases by looking through the images/ folder, put the images you think that are good enough to the good/ folder. 
   the cases in the good/ folder will never be processed again.
3. while there are bad cases left, identify common problems, like (a) extra skull left; (b) grey matter being cut by BET, or other problems
4. create a folder (folder name to indicate the problem type) and put the cases's images with similar problem into this folder 
6. change the "mode" in pipeline_config.json to be the same as the new folder name, 
   adjust the parameters in the pipeline_config.json to address the problems of this type of problem.
5. run the main.py again to batch-wise process all cases with this common problem
6. review the outcome from the images/ folder, put good cases in good/ folder
7. loop back to step 4 until all cases are in the good/ folder

folder structure:
scans/               save the most updated nifit scans 
images/              save the most updated image 
tmp/                 tmp working folder
good/                a pool of properly processed cases
"""


