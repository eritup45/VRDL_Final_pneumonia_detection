import os
import time

import skimage 
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import PIL

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as torchDataset
import torchvision as tv
from torch.autograd import Variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

import shutil
import pydicom
#################

import matplotlib.pyplot as plt
import pylab
import numpy as np
import pydicom
import pandas as pd
from glob import glob
import os
from matplotlib.patches import Rectangle
import cv2



def display_image_per_patient(df, pId, angle=0.0, sample='train'):
    '''
    Given one patient ID and the dataset,
    display the corresponding dicom image with overlaying boxes and class annotation.
    To be implemented: Optionally input the image rotation angle, in case of data augmentation.
    '''
    dcmdata = get_dcm_data_per_patient_old(pId, sample=sample)
    dcmimg = dcmdata.pixel_array
    boxes = get_boxes_per_patient(df, pId)
    plt.figure(figsize=(20,10))
    plt.imshow(dcmimg, cmap=pylab.cm.binary)
    
    class_color_dict = {'Normal' : 'green',
                        'No Lung Opacity / Not Normal' : 'orange',
                        'Lung Opacity' : 'red'}

    if len(boxes)>0:
        for box in boxes:
            # extracting individual coordinates and labels
            x, y, w, h, c, t = box 
            # create a rectangle patch, (x,y) is upper left 
            patch = Rectangle((x,y), w, h, color='red', 
                              angle=angle, fill=False, lw=4, joinstyle='round', alpha=0.6)
            # get current axis and draw rectangle
            plt.gca().add_patch(patch)
            
    # add annotation text
    plt.text(10, 50, c, color=class_color_dict[c], size=20, 
             bbox=dict(edgecolor=class_color_dict[c], facecolor='none', alpha=0.5, lw=2))

def dcm2png(pIds, inputdir = './data/stage_2_train_images/', outdir = './train/'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for p in pIds:
        ds = get_dcm_data_per_patient(p)
        img = ds.pixel_array # get image array
        print(f"{inputdir + p + '.dcm'} -> {outdir + p + '.png'}")
        cv2.imwrite(outdir + p + '.png', img) # write png image

# Convert .dcm to .png images
def folder_dcm2png(inputdir = './data/stage_2_test_images/', outdir = './test/'):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    test_list = [ f for f in  os.listdir(inputdir)]
    for f in test_list:   # remove "[:10]" to convert all images 
        ds = pydicom.read_file(inputdir + f) # read dicom image
        img = ds.pixel_array # get image array
        print(f"{inputdir + f} -> {outdir + f.replace('.dcm', '.png')}")
        cv2.imwrite(outdir + f.replace('.dcm','.png'), img) # write png image

def get_boxes_per_patient(df, pId):
    '''
    Given the dataset and one patient ID, 
    return an array of all the bounding boxes and their labels associated with that patient ID.
    Example of return: 
    array([[x1, y1, width1, height1, class1, target1],
           [x2, y2, width2, height2, class2, target2]])
    '''
    
    boxes = df.loc[df['patientId']==pId][['x', 'y', 'width', 'height', 'class', 'Target']].values
    return boxes

def get_dcm_data_per_patient_old(pId, sample='train', datapath = './data/'):
    '''
    Given one patient ID and the sample name (train/test), 
    return the corresponding dicom data.
    '''
    return pydicom.read_file(datapath+'stage_2_'+sample+'_images/'+pId+'.dcm')

def get_dcm_data_per_patient(pId, inputdir = './data/stage_2_train_images/'):
    '''
    Given one patient ID and the folder (train/test), 
    return the corresponding dicom data.
    '''
    return pydicom.read_file(inputdir+pId+'.dcm')

def get_png_name_per_patient(pId, sample='train', datapath = './data/'):
    '''
    Given one patient ID and the sample name (train/test), 
    return the corresponding dicom data.
    '''
    return (datapath+'stage_2_'+sample+'_images/'+pId+'.png')

def csv2yolo(df, pIds_list, outdir = './val/'):
    for pId in pIds_list:
        with open(outdir + pId + '.txt', 'w') as fp:
            ds = get_dcm_data_per_patient(pId, inputdir = './data/stage_2_train_images/')
            img = ds.pixel_array
            img_width, img_height = img.shape # 2d
            boxes = get_boxes_per_patient(df, pId)
            box_l = len(boxes)
            for i, box in enumerate(boxes):
                x, y, w, h, c, t = box 
                if t == 0:
                    break
                else:
                    label = c
                    # Normalize (yolov5)
                    x_center = ((x + w) / 2) / img_width
                    y_center = ((y + h) / 2) / img_height
                    bbox_width = w / img_width
                    bbox_height = h / img_height
                    s = str(label)+' '+str(x_center)+' '+str(y_center)+' '+str(bbox_width)+' '+str(bbox_height)
                    if i!=(box_l-1):
                        s += '\n'
                    # print(s)
                    fp.write(s)

if __name__ == '__main__':
    # folder_dcm2png() # one time should be enough
    datapath = './data/'

    # stage_2_train_labels
    df_box = pd.read_csv(datapath+'stage_2_train_labels.csv')
    # stage_2_detailed_class_info.csv
    df_aux = pd.read_csv(datapath+'stage_2_detailed_class_info.csv')
    # Merge (Same length)
    df_all = pd.concat([df_box, df_aux.drop(labels=['patientId'], axis=1)], axis=1)
    df_all.to_csv('train.csv') # write .csv file

    # Bounding Boxes and Target Label Data
    print('Number of rows (unique boxes per patient) in main train dataset:', df_all.shape[0])
    print('Number of unique patient IDs:', df_all['patientId'].nunique())

    validation_frac = 0.1
    # df_all = df_all.sample(frac=1, random_state=42) # .sample(frac=1) does the shuffling
    pIds = [pId for pId in df_all['patientId'].unique()]
    pIds_valid = pIds[ : int(round(validation_frac*len(pIds)))]
    pIds_train = pIds[int(round(validation_frac*len(pIds))) : ]
    print('{} patient IDs shuffled and {}% of them used in validation set.'.format(len(pIds), validation_frac*100))
    print('{} images went into train set and {} images went into validation set.'.format(len(pIds_train), len(pIds_valid)))

    # remove comment to reorganize images
    # dcm2png(pIds_valid, inputdir = './data/stage_2_train_images/', outdir = './val/')
    # dcm2png(pIds_train, inputdir = './data/stage_2_train_images/', outdir = './train/')
    # folder_dcm2png(inputdir = './data/stage_2_test_images/', outdir = './test/')

    df_val = df_all.loc[df_all['patientId'].isin(pIds_valid)]
    df_train = df_all.loc[df_all['patientId'].isin(pIds_train)]
    # remove comment if saving csv
    # df_train.to_csv('./train/train.csv')
    # df_val.to_csv('./val/val.csv')

    # remove comment if saving yolo label files
    # csv2yolo(df_val, pIds_valid, outdir = './val/')
    # csv2yolo(df_train, pIds_train, outdir = './train/')

    print('----------------df_all[\'Target\']==1-----------------')
    df_all = df_all.loc[df_all['Target']==1]
    print('Number of rows (unique boxes per patient) in main train dataset:', df_all.shape[0])
    print('Number of unique patient IDs:', df_all['patientId'].nunique())

    validation_frac = 0.1
    # df_all = df_all.sample(frac=1, random_state=42) # .sample(frac=1) does the shuffling
    pIds = [pId for pId in df_all['patientId'].unique()]
    pIds_valid = pIds[ : int(round(validation_frac*len(pIds)))]
    pIds_train = pIds[int(round(validation_frac*len(pIds))) : ]
    print('{} patient IDs shuffled and {}% of them used in validation set.'.format(len(pIds), validation_frac*100))
    print('{} images went into train set and {} images went into validation set.'.format(len(pIds_train), len(pIds_valid)))
    
    # remove comment to reorganize images
    # dcm2png(pIds_valid, inputdir = './data/stage_2_train_images/', outdir = './val_confirm/')
    # dcm2png(pIds_train, inputdir = './data/stage_2_train_images/', outdir = './train_confirm/')
    # folder_dcm2png(inputdir = './data/stage_2_test_images/', outdir = './test/')

    df_val = df_all.loc[df_all['patientId'].isin(pIds_valid)]
    df_train = df_all.loc[df_all['patientId'].isin(pIds_train)]
    # df_train.to_csv('./train/train_confirm.csv')
    # df_val.to_csv('./val/val_confirm.csv')

    # remove comment if saving yolo label files
    csv2yolo(df_val, pIds_valid, outdir = './val_confirm/')
    csv2yolo(df_train, pIds_train, outdir = './train_confirm/')



