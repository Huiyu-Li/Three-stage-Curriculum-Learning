from medpy import metric
import numpy as np
from scipy import ndimage
import time
from surface import Surface
LARGE = 9001

def dice(input1, input2):
    return metric.dc(input1, input2)

def TP(pred,ref,threshold):
    r_id_list = np.unique(ref)
    if r_id_list[0] == 0:
        r_id_list = r_id_list[1:]
    num_pred = 0
    roi = np.logical_and(pred, ref)
    for i in r_id_list:
        ref_i = (ref==i)
        bounding_box = ndimage.find_objects(ref_i)[0]
        roi_i = roi[bounding_box]
        overlap = sum(sum(sum(roi_i)))/sum(sum(sum(ref_i)))#need to pay attention
        if overlap>=threshold:
            num_pred+=1
    return num_pred

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

def compute_tumor_burden(prediction_mask, reference_mask):
    """
    Calculates the tumor_burden and evalutes the tumor burden metrics RMSE and
    max error.

    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :return: dict with RMSE and Max error
    """

    def calc_tumor_burden(vol):
        num_liv_pix = np.count_nonzero(vol >= 1)
        num_les_pix = np.count_nonzero(vol == 2)
        return num_les_pix / float(num_liv_pix)

    tumor_burden_r = calc_tumor_burden(reference_mask)
    if np.count_nonzero(prediction_mask == 1):
        tumor_burden_p = calc_tumor_burden(prediction_mask)
    else:
        tumor_burden_p = LARGE

    tumor_burden_diff = tumor_burden_r - tumor_burden_p
    return tumor_burden_diff

def test():
    pred_mask_lesion = np.array([[1, 0, 0, 2],
                                 [0, 0, 0, 0],
                                 [3, 3, 3, 3],
                                 [3, 3, 0, 0]])
    ref_mask_lesion = np.array([[1, 0, 0, 0],
                                [0, 0, 2, 2],
                                [0, 0, 2, 2],
                                [0, 0, 0, 0]])
    # detected_mask, mod_reference_mask, num_detected = detect_lesions(pred_mask_lesion, ref_mask_lesion, min_overlap=0.5)
    # print(detected_mask)
    # print(mod_reference_mask)
    # print(num_detected)


    TP(pred_mask_lesion, ref_mask_lesion, 0.5)


# test()