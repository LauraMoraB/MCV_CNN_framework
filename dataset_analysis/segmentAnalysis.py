import numpy as np
import os
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

classes_camvid = {
        0: 'sky',
        1: 'building',
        2: 'column_pole',
        3: 'road',
        4: 'sidewalk',
        5: 'tree',
        6: 'sign',
        7: 'fence',
        8: 'car',
        9: 'pedestrian',
        10: 'byciclist',
        11: 'void'
}

classes_kitti = {
        0: 'sky',
        1: 'building',
        2: 'column_pole',
        3: 'road',
        4: 'sidewalk',
        5: 'tree',
        6: 'sign',
        7: 'fence',
        8: 'car',
        9: 'pedestrian',
        10: 'byciclist',
        11: 'void'
}

classes_synthia = {
        0:  'road',
        1:  'sidewalk',
        2:  'building',
        3:  'wall',
        4:  'fence',
        5:  'pole',
        6:  'traffic light',
        7:  'traffic sign',
        8:  'vegetation',
        9:  'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        19: 'void'
    }

def analyzeImage(image, class_dict):
    num_pixels = np.array([0]*len(class_dict))
    for index,name in class_dict.items():
        num_pixels[index] += np.count_nonzero(image==(index,index,index))
    return num_pixels

def analyzeFolder(path,class_dict):
    fileList = os.listdir(path)
    class_px_count = np.array([0]*len(class_dict), dtype=np.float)
    n_images = len(fileList)
    count_apparitions = np.zeros(len(class_dict))
    for gt_name in fileList:
        # print(path+gt_name)
        img = cv.imread(path+gt_name)
        class_px_count += analyzeImage(img,class_dict)
        count_apparitions += class_px_count != 0
    return class_px_count, count_apparitions, n_images

def plotStats(class_px_count,class_dict,imageName):
    class_labels = list(class_dict.values())
    plt.pie(class_px_count, labels=class_labels)
    plt.axis('equal')
    plt.savefig(imageName)

def printStats(class_px_count,class_dict, count_apparitions, n_images):
    total_px = np.sum(class_px_count)
    percent_array = (class_px_count/total_px)*100

    class_labels = list(class_dict.values())
    print("-------------------")
    print("Class stats in %:")
    print("")

    str_sz = max(len(x) for x in class_labels)
    for index,item_label in enumerate(class_labels):
        this_sz = len(item_label)
        remaining_space = str_sz - this_sz
        white_space = "".join([" "]*remaining_space)
        p_painted = '%.2f' % percent_array[index]
        p_apparit = '%.2f' % (count_apparitions[index]/n_images*100)
        string = "{}{} \t {} \t {}/{}={}".format(item_label, white_space, p_painted, int(count_apparitions[index]), n_images, p_apparit)
        print(string )
    print("-------------------")




# img = cv.imread("/home/grupo08/M5/dataset/segmentation/camvid/test/masks/Seq05VD_f00930.png")
# num_pixels = analyzeImage(img,classes_camvid)
# print(num_pixels)

#num_pixels, count_apparitions, n_images = analyzeFolder("/home/grupo08/M5/dataset/segmentation/camvid/valid/masks/",classes_camvid)
# num_pixels, count_apparitions, n_images = analyzeFolder("/home/grupo08/M5/dataset/segmentation/kitti/train/masks/",classes_kitti)
#num_pixels, count_apparitions, n_images = analyzeFolder("/home/grupo08/M5/dataset/segmentation/synthia_rand_cityscapes/test/masks/",classes_synthia)
#num_pixels, count_apparitions, n_images = analyzeFolder("/home/grupo08/M5/executions/sem_seg_camvid/sem_seg_camvid/predictions/",classes_camvid)
#num_pixels, count_apparitions, n_images = analyzeFolder("/home/grupo08/M5/executions/sem_seg_kitty/CamVid_SemSeg_test1_kitty/predictions/", classes_kitti)

print(num_pixels)
print("APPARITION:", count_apparitions)
printStats(num_pixels,classes_kitti, count_apparitions, n_images)
plotStats(num_pixels,classes_kitti,"./plotPREDICTIONS_Kitti.png")

















#
