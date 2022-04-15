this folder contains the scripts to generate shap inter-region correlation graph

This folder should already contain the following files:
network_axi_combine_regions.csv
network_sag_combine_regions.csv
Hammers_mith_atlases_n30r95_label_indices_SPM12_20170315.xml
plot_sagittal_network.py
plot_axial_network.py
glassbrain_background_axial.png
glassbrain_background_sagittal.png

to use the code, you need to firstly generate 2 numpy arrays as described below in the current folder:
regional95_avgScore_ADD.npy (should be of size (N, 95), N is the number of ADD subjects,
                             95 is the number of regions from the Adult Brain Atlas, each
                             element [i][j] in the numpy array represents the shap value
                             of that subject[i] for the region[j].)
regional95_avgScore_nADD.npy (should be of size (N, 95), N is the number of non-ADD subjects,
                             95 is the number of regions from the Adult Brain Atlas, each
                             element [i][j] in the numpy array represents the shap value
                             of that subject[i] for the region[j].)

Then run the scirpts below will generate the figures in the plot/ folder.
(if plot/ folder doesn't exist, please create am empty folder named as plot before running the script below)
plot_sagittal_network.py
plot_axial_network.py

