from scipy.ndimage.measurements import label
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
import os
import time
import shutil
from datetime import datetime

# Generate subimage by barycenter
def generate_subimage(ct_array,seg_array,stridez, stridex, stridey, blockz, blockx, blocky,
					  idx,origin,direction,xyz_thickness,savedct_path,savedseg_path):
	num_z = (ct_array.shape[0]-blockz)//stridez + 1#math.floor()
	num_x = (ct_array.shape[1]-blockx)//stridex + 1
	num_y = (ct_array.shape[2]-blocky)//stridey + 1
	for z in range(num_z):
		for x in range(num_x):
			for y in range(num_y):
				seg_block = seg_array[z*stridez:z*stridez+blockz,x*stridex:x*stridex+blockx,y*stridey:y*stridey+blocky]
				if seg_block.any():
					ct_block = ct_array[z * stridez:z * stridez + blockz, x * stridex:x * stridex + blockx,
							   y * stridey:y * stridey + blocky]
					saved_ctname = os.path.join(savedct_path, 'Novolume-' + str(idx) + '.nii')
					saved_segname = os.path.join(savedseg_path, 'Nosegmentation-' + str(idx) + '.nii')
					saved_preprocessed(ct_block,origin,direction,xyz_thickness,saved_ctname)
					saved_preprocessed(seg_block,origin,direction,xyz_thickness,saved_segname)
					idx = idx + 1
	return idx

def saved_preprocessed(savedImg,origin,direction,xyz_thickness,saved_name):
	newImg = sitk.GetImageFromArray(savedImg)
	newImg.SetOrigin(origin)
	newImg.SetDirection(direction)
	newImg.SetSpacing((xyz_thickness[0], xyz_thickness[1], xyz_thickness[2]))
	sitk.WriteImage(newImg, saved_name)

def preprocess():
    start_time = time.time()
    ##########hyperparameters1##########
    f2 = open("./NoneTumor.txt", "r")
    lines = f2.readlines()
    num = len(lines)
    ct_path = '/media/lihuiyu/sda4/LITS/Preprocessed3/ct'
    seg_path = '/media/lihuiyu/sda4/LITS/Preprocessed3/seg'
    savedct_path = '/media/lihuiyu/sda4/LITS/NoneTumorPatches/ct'
    savedseg_path = '/media/lihuiyu/sda4/LITS/NoneTumorPatches/seg'

    blockz = 64;blockx = 190;blocky = 190
    stridez = blockz//3;stridex = blockx//3;stridey = blocky//3
    saved_idx = 0
    ##########end hyperparameters1##########
    # Clear saved dir
    if os.path.exists(savedct_path) is True:
        shutil.rmtree(savedct_path)
    os.mkdir(savedct_path)
    if os.path.exists(savedseg_path) is True:
        shutil.rmtree(savedseg_path)
    os.mkdir(savedseg_path)

    for i in range(num):#num
        ct = sitk.ReadImage(os.path.join(ct_path, lines[i].strip('\n')).replace('segmentation','volume'))
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        spacing = ct.GetSpacing()
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(seg_path, lines[i].strip('\n'))))

        # step4:generate subimage
        # step5 save the preprocessed data
        saved_idx = generate_subimage(ct_array,seg_array,stridez, stridex, stridey, blockz, blockx, blocky,
					  saved_idx,origin,direction,spacing,savedct_path,savedseg_path)

        print('Time {:.3f} min'.format((time.time() - start_time) / 60))
        print(saved_idx)


if __name__ == '__main__':
    preprocess()
