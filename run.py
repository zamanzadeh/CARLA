# %%
import numpy as np
import pandas as pd
import os
import subprocess

# %%
# with open('/home/zahraz/hz18_scratch/zahraz/datasets/MSL_SMAP/labeled_anomalies.csv', 'r') as file:
#     csv_reader = pd.read_csv(file, delimiter=',')
# data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']

all_files = os.listdir(os.path.join('/home/zahraz/hz18_scratch/zahraz/datasets/', 'SMD/train'))
file_list = [file for file in all_files if file.startswith('machine-')]
file_list = sorted(file_list)
print(file_list)

# file_list = os.listdir(os.path.join('/home/zahraz/hz18_scratch/zahraz/datasets/', 'UCR'))                        
# file_list = sorted(file_list)

# for filename in files: #['swat']: #files: #data_info['chan_id']:
#     if filename != 'GECCO':
#         print(filename)
        
#         # Run the pretext script
#         subprocess.run([
#             'python', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/carla_pretext.py',
#             '--config_env', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/env.yml',
#             '--config_exp', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/pretext/carla_pretext_smd.yml',
#             '--fname', filename
#         ])
        
#         # Run the classification script
#         subprocess.run([
#             'python', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/carla_classification.py',
#             '--config_env', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/env.yml',
#             '--config_exp', '/home/zahraz/hz18_scratch/zahraz/Published/CARLA/configs/classification/carla_classification_smd.yml',
#             '--fname', filename
#         ])
index = file_list.index("machine-3-11.txt")

for filename in file_list: #[index:]:  #['GECCO']: #['machine-2-4.txt']:
    # if 'real_' in filename:
    if filename != 'GECCO':
        print(filename)
        genmodel = 'gen_anom_' + filename+'.pth'

        # Run the pretext script
        subprocess.run([
            'python', 'carla_pretext.py',
            '--config_env', 'configs/env.yml',
            '--config_exp', 'configs/pretext/carla_pretext_smd.yml',
            '--fname', filename
        ], check=True)
        
        # Run the classification script
        subprocess.run([
            'python', 'carla_classification.py',
            '--config_env', 'configs/env.yml',
            '--config_exp', 'configs/classification/carla_classification_smd.yml',
            '--fname', filename
        ], check=True)


