from scipy.ndimage.measurements import label
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
import os
import time
import shutil
from datetime import datetime

# 1. write the case with and without tumor into txt
# 2. Calculating the patch size of tumor
def FindTumor():
    start_time = time.time()
    ##########hyperparameters##########
    #please add the dataset name as the prefix
    savedct_path = '/media/lihuiyu/sda3/KiTS/Preprocessed/ct/'
    savedseg_path = '/media/lihuiyu/sda3/KiTS/Preprocessed/seg/'
    tumorlog = './KiTSTumor0.txt'
    Ntumorlog = './KiTSNoneTumor0.txt'
    sizelog = './KiTSTumorSize0.txt'
    ##########end hyperparameters1##########
    file_list = os.listdir(savedseg_path)
    num_file = len(file_list)

    if os.path.isfile(tumorlog):
        os.remove(tumorlog)
    else:
        open(tumorlog, 'w')
    if os.path.isfile(Ntumorlog):
        os.remove(Ntumorlog)
    else:
        open(Ntumorlog, 'w')

    if os.path.isfile(sizelog):
        os.remove(sizelog)
    else:
        with open(sizelog, 'w') as log:
            log.write(str(datetime.now()) + '\n')
    sizelist = []
    for i in range(30): #num_file
        seg = sitk.ReadImage(os.path.join(savedseg_path,file_list[i]), sitk.sitkFloat32)
        seg_array = sitk.GetArrayFromImage(seg)
        if (seg_array==2).any():
            print(file_list[i])
            with open(tumorlog, 'a') as log1:
                log1.write(os.path.join(file_list[i]) + '\n')#write the case with tumor into txt
                # Calculating the patch size of tumor
                with open(sizelog, 'a') as log3:
                    mask_lesion, num_predicted = label(seg_array == 2, output=np.int16)
                    id_list = np.unique(mask_lesion)
                    if id_list[0] == 0:
                        id_list = id_list[1:]
                    print(id_list)
                    for id in id_list:
                        one = np.where(mask_lesion == id)[0].max() - np.where(mask_lesion == id)[0].min()
                        two = np.where(mask_lesion == id)[1].max() - np.where(mask_lesion == id)[1].min()
                        three = np.where(mask_lesion == id)[2].max() - np.where(mask_lesion == id)[2].min()
                        temp = [one, two, three]
                        sizelist.append(temp)
                        log3.write(str(one) + ',' + str(two) + ',' + str(three) + '\n')
        else:
            print(file_list[i])
            with open(Ntumorlog, 'a') as log2:
                log2.write(os.path.join(file_list[i]) + '\n')#write the case without tumor into txt
    # Calculating the patch size of tumor
    sizearray = np.array(sizelist)
    num_tumor = sizearray.shape;print('num_tumor:',num_tumor)
    max_0 = sizearray[:, 0].max()
    max_1 = sizearray[:, 1].max()
    max_2 = sizearray[:, 2].max()
    print('pathch_size:','[',max_0,',',max_1,',',max_2,']')
    with open(sizelog, 'a') as log3:
        log3 .write('num_tumor:'+str(num_tumor)+'\n')
        log3.write('pathch_size:'+'['+str(max_0) + ',' + str(max_1) + ',' + str(max_2) +']'+ '\n')
    print('Time elapsed:', (time.time() - start_time) // 60)

# Generate subimage by barycenter
def generate_subimage(ct_array,seg_array,blockz, blockx, blocky,
					  idx,origin,direction,xyz_thickness,savedct_path,savedseg_path):
    stridez = blockz//2; stridex = blockx//2;  stridey = blocky//2;
    label_lesion = label(seg_array == 2, output=np.int16)[0]
    id_list = np.unique(label_lesion)
    if id_list[0] == 0:
        id_list = id_list[1:]
    print(id_list)
    barycenter = ndimage.measurements.center_of_mass(seg_array, label_lesion, id_list)
    for coord in barycenter:
        print(coord)
        zmin = max(0,int(coord[0])-stridez)
        zmax = min(64,int(coord[0])+stridez)
        if zmin == 0:
            zmax = blockz
        if zmax == 64:
            zmin = 64-blockz

        xmin = max(0, int(coord[1]) - stridex)
        xmax = min(256, int(coord[1]) + stridex)
        if xmin == 0:
            xmax = blockx
        if xmax == 256:
            xmin = 256 - blockx

        ymin = max(0, int(coord[2]) - stridey)
        ymax = min(256, int(coord[2]) + stridey)
        if ymin == 0:
            ymax = blocky
        if ymax == 256:
            ymin = 256 - blocky

        seg_block = seg_array[zmin:zmax, xmin:xmax, ymin:ymax]
        ct_block = ct_array[zmin:zmax, xmin:xmax, ymin:ymax]

        saved_ctname = os.path.join(savedct_path, 'volume-' + str(idx) + '.nii')
        saved_segname = os.path.join(savedseg_path, 'segmentation-' + str(idx) + '.nii')
        saved_preprocessed(ct_block, origin, direction, xyz_thickness, saved_ctname)
        saved_preprocessed(seg_block, origin, direction, xyz_thickness, saved_segname)
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
    f2 = open("./Tumor.txt", "r")
    lines = f2.readlines()
    num = len(lines)
    ct_path = '/media/lihuiyu/sda4/LITS/Preprocessed3/ct'
    seg_path = '/media/lihuiyu/sda4/LITS/Preprocessed3/seg'
    savedct_path = '/media/lihuiyu/sda4/LITS/TumorPatches/ct'
    savedseg_path = '/media/lihuiyu/sda4/LITS/TumorPatches/seg'

    blockz = 64;blockx = 190;blocky = 190
    # stridez = blockz//3;stridex = blockx//3;stridey = blocky//3
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
        saved_idx = generate_subimage(ct_array, seg_array, blockz, blockx, blocky,
                          saved_idx, origin, direction,spacing,savedct_path,savedseg_path)

        print('Time {:.3f} min'.format((time.time() - start_time) / 60))
        print(saved_idx)


if __name__ == '__main__':
    # preprocess()
    FindTumor()