#!/usr/bin/env python
#Copy from evaluate
from __future__ import print_function
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
import gc
import pandas as pd
from scipy import ndimage
from medpy import metric
from surface import Surface
import time
LARGE = 1
def dice(input1, input2):
    return metric.dc(input1, input2)
def compute_segmentation_scores(prediction_mask, reference_mask, voxel_spacing):
    """
    Calculates metrics scores from numpy arrays and returns an dict.
    Assumes that each object in the input mask has an integer label that
    defines object correspondence between prediction_mask and
    reference_mask.
    :param prediction_mask: numpy.array, int
    :param reference_mask: numpy.array, int
    :param voxel_spacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voe, rvd, assd, rmsd, and msd
    """
    scores = {'dice': [],
              'jaccard': [],
              'voe': [],
              'rvd': [],
              'assd': [],
              'rmsd': [],
              'msd': []}

    p = (prediction_mask>0)
    r = (reference_mask>0)
    if np.any(p) and np.any(r):
        dice = metric.dc(p, r)
        jaccard = dice / (2. - dice)
        scores['dice'].append(dice)
        scores['jaccard'].append(jaccard)
        scores['voe'].append(1. - jaccard)
        scores['rvd'].append(metric.ravd(r, p))
        evalsurf = Surface(p, r,
                           physical_voxel_spacing=voxel_spacing,
                           mask_offset=[0., 0., 0.],
                           reference_offset=[0., 0., 0.])

        assd = evalsurf.get_average_symmetric_surface_distance()
        rmsd = evalsurf.get_root_mean_square_symmetric_surface_distance()
        msd = evalsurf.get_maximum_symmetric_surface_distance()
        scores['assd'].append(assd)
        scores['rmsd'].append(rmsd)
        scores['msd'].append(msd)
    elif not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
        scores['dice'].append(1)
        scores['jaccard'].append(1)
        scores['voe'].append(0)
        scores['rvd'].append(0)
        scores['assd'].append(0)
        scores['rmsd'].append(0)
        scores['msd'].append(0)
    else:
        # There are no objects in the prediction, in the reference, or both
        scores['dice'].append(0)
        scores['jaccard'].append(0)
        scores['voe'].append(1.)
        # Surface distance (and volume difference) metrics between the two
        # masks are meaningless when any one of the masks is empty. Assign
        # maximum penalty. The average score for these metrics, over all
        # objects, will thus also not be finite as it also loses meaning.
        scores['rvd'].append(LARGE)
        scores['assd'].append(LARGE)
        scores['rmsd'].append(LARGE)
        scores['msd'].append(LARGE)
    return scores

import re
def atoi(s):
    return int(s) if s.isdigit() else s
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

start_time = time.time()
segmentation_metrics = {'dice': 0,
                        'jaccard': 0,
                        'voe': 1,
                        'rvd': LARGE,
                        'assd': LARGE,
                        'rmsd': LARGE,
                        'msd': LARGE}
# Initialize results dictionaries
lesion_segmentation_scores = {}
dice_per_case = {'lesion': []}
dice_global_x = {'lesion': {'I': 0, 'S': 0}} # 2*I/S
# valide for three-stage curriculum learning
# pred_valid_path = "/home/lihuiyu/Data/LiTS/segResults/20191002-771valid/"
# true_valid_path = "/media/lihuiyu/sda4/LITS/Preprocessed3/seg/"
# valide for Whole-to-Patch Curriculum Training
# pred_valid_path = "/home/lihuiyu/Data/LiTS/segResults/20191004-213440-Whole-to-Patch/"
# true_valid_path = "/media/lihuiyu/sda4/LITS/Preprocessed3/seg/"
# valide for Naive Training
# pred_valid_path = "/home/lihuiyu/Data/LiTS/segResults/20191004-222915-Naive/"
pred_valid_path = "/home/lihuiyu/Data/LiTS/segResults/20191005-085213-CascadeLiver-and-NaiveTumor/"
true_valid_path = "/media/lihuiyu/sda4/LITS/Preprocessed3/seg/"
# valide for Patch-to-Whole Curriculum Training
# pred_valid_path = "/home/lihuiyu/Data/LiTS/segResults/20191004-230404-Patch-to-Whole/"
# true_valid_path = "/media/lihuiyu/sda4/LITS/Preprocessed3/seg/"


file_lists = os.listdir(pred_valid_path)
file_lists.sort(key=natural_keys)
for name in file_lists:
    if name.endswith('output2.nii'):
        print(name)
        true_name = 'segmentation-' + name.split('-')[2] + '.nii'
        pred = sitk.ReadImage(os.path.join(pred_valid_path, name))
        voxel_spacing = pred.GetSpacing()

        pred_mask_lesion = sitk.GetArrayFromImage(pred).astype(np.int8)
        true_mask_lesion = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(true_valid_path, true_name))).astype(np.int8)
        true_mask_lesion = (true_mask_lesion==2)
        # Compute lesions segmentation scores.
        lesion_scores = compute_segmentation_scores(
                                          prediction_mask=pred_mask_lesion,
                                          reference_mask=true_mask_lesion,
                                          voxel_spacing=voxel_spacing)
        for metric in segmentation_metrics:
            if metric not in lesion_segmentation_scores:
                lesion_segmentation_scores[metric] = []
            lesion_segmentation_scores[metric].extend(lesion_scores[metric])

        # Compute per-case (per patient volume) dice.
        from medpy import metric
        if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
            dice_per_case['lesion'].append(1.)
        else:
            dice_per_case['lesion'].append(metric.dc(pred_mask_lesion,true_mask_lesion))

        # Accumulate stats for global (dataset-wide) dice score.
        dice_global_x['lesion']['I'] += np.count_nonzero(np.logical_and(pred_mask_lesion, true_mask_lesion))
        dice_global_x['lesion']['S'] += np.count_nonzero(pred_mask_lesion) + np.count_nonzero(true_mask_lesion)

gc.collect()

#####################################################################
# Compute lesion segmentation metrics.
lesion_segmentation_metrics = {}
for m in lesion_segmentation_scores:
    lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
if len(lesion_segmentation_scores)==0:
    # Nothing detected - set default values.
    lesion_segmentation_metrics.update(segmentation_metrics)
lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
dice_global = 2.*dice_global_x['lesion']['I']/dice_global_x['lesion']['S']
lesion_segmentation_metrics['dice_global'] = dice_global

# Print results to stdout.
print("####Computed leision segmentation metrics (for detected lesions):")
for metric, value in lesion_segmentation_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))

# Write metrics to file.
output_filename = os.path.join('./', 'scores.txt')
output_file = open(output_filename, 'w')
for metric, value in lesion_segmentation_metrics.items():
    output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))

output_file.close()

print('Time {:.3f} min'.format((time.time() - start_time) / 60))
print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))