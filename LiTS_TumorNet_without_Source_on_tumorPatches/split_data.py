##########Split data into epi-train-valid-test##########
import csv
import math
import numpy as np
import os
import shutil
import re
def atoi(s):
    return int(s) if s.isdigit() else s
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

##########hyperparameters##########
#LiTS
savedct_path = '/media/lihuiyu/sda4/LITS/TumorPatches/ct'
savedseg_path = '/media/lihuiyu/sda4/LITS/TumorPatches/seg'

TVTcsv = './TVTcsv_tumor'
valid_csv = 'valid_tumor.csv'
# test_csv = 'test.csv'
episode = 10
ratio = 0.95
##########end hyperparameters##########
ct_lists = os.listdir(savedct_path)
ct_lists.sort(key=natural_keys)
total = len(ct_lists)

tn = math.ceil(total*ratio)
tn_epi = tn//episode
tn = tn_epi*episode#remove the train tail
valid_lists = ct_lists[tn:total]

#clear the exists file
if os.path.isdir(TVTcsv):
    shutil.rmtree(TVTcsv)
os.mkdir(TVTcsv)

train_csv_list = ['train'+str(i)+'.csv' for i in range(episode)]
for epi in range(episode):
    train_lists = ct_lists[epi * tn_epi:(epi + 1) * tn_epi]#attention:[0:num_train)
    with open(os.path.join(TVTcsv,train_csv_list[epi]), 'w') as file:
        w = csv.writer(file)
        for name in train_lists:
            ct_name = os.path.join(savedct_path, name)
            seg_name = os.path.join(savedseg_path, 'segmentation-' + name.split('-')[-1])
            w.writerow((ct_name, seg_name))#attention: the first row defult to tile

with open(os.path.join(TVTcsv,valid_csv), 'w') as file:
    w = csv.writer(file)
    for name in valid_lists:
        ct_name = os.path.join(savedct_path, name)
        seg_name = os.path.join(savedseg_path, 'segmentation-' + name.split('-')[-1])
        w.writerow((ct_name,seg_name))
print('total=',total,'train=',tn,'(',tn_epi,'*',episode,')', 'val=',len(valid_lists))
