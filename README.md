# Three-stage-Curriculum-Training-for-Tumor-Segmentation

## 0. Introduction
This repository contains Pytorch code for the paper entitled with"A New Curriculum Learning Approach to Deep Network Based Liver Tumor Segmentation" . This paper was initially described in arXiv (https://arxiv.org/abs/1910.07895). 
## 1. Getting Started
Clone the repo:  https://github.com/Huiyu-Li/Three-stage-Curriculum-Learning.git
#### Requirements
~~~
python>=3.6
torch>=0.4.0
torchvision
csv
pandas
json
scipy
SimpleITK
medpy
numpy
time
shutil
sys
os
~~~
## 2. Data Prepare
   You need to have downloaded at least the LiTS 2017 training dataset.
   First, you are supposed to make a dataset directory.
   Second, you may need to preprocess the data by  https://github.com/Huiyu-Li/Preprocess-of-CT-data
   Third, change the file path in the **hyperparameters** part in the Main.py
## 3. Usage
### To train the model:
####  • Stage 1: 
Step1: split the data into training and valid dataset, respectively.
LiTS_TumorNet_without_Source _on_wholeData>split_data.py 
Step2: Training
~~~
##########hyperparameters##########
if_test = False
if_resume = False# changed as True if you have saved model
##########hyperparameters##########
~~~
#### • Stage 2: 
Step1: Extract tumor patches form the whole input
GetTumorPathes>LiTSGetNegtiveTumorPatches.py and LiTSGetPositiveTumorPatches.py
Step2: split the data into training and valid dataset, respectively.
LiTS_TumorNet_without_Source_on_tumorPatches>split_datawithNegtive.py 
Step3: Training
~~~
##########hyperparameters##########
if_test = False
if_resume = True
##########hyperparameters##########
~~~
#### • Stage 3: 
Just like the Stage 1.
~~~
##########hyperparameters##########
if_test = False
if_resume = True
##########hyperparameters##########
~~~
### To Test and evaluate model:
   Step1: 
~~~
##########hyperparameters##########
if_test = True
if_resume = True
##########hyperparameters##########
~~~
   Step2: LiTS_Evaluation>evaluator1.py
