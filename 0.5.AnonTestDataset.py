# This script will generate a copy of the test dataset with anonomized folder names. 
# Data output for inference will depend on the folder name and must therefor be anonomized. 
# Author: Christian Jamtheim Gustafsson, PhD

import os
import cv2
import numpy as np
import os.path
import pydicom
import scipy
import nibabel as nib
from random import seed
from random import randint
import shutil
import pydicom
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
from sharedFunctions import * 
from distutils.dir_util import copy_tree

# Define what dataset to process
procDataSet = 'ProstateDataBigRawSortedTestDataset'
# Patient folder
allPatFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/' + procDataSet + '/'
# List patients in patFolder
patFolders = os.listdir(allPatFolder)
# Define output folder 
outFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/' + procDataSet + 'Anon'

# Make sure the output directory exists
if not os.path.isdir(outFolder):
    # Create dir
    os.mkdir(outFolder)

for patNr, patient in enumerate(patFolders):
    # Create new random seed
    # Important for parallell threading
    R=np.random.RandomState()
    # Create random large integer (this is not overlapping with training data numbers)
    # Use it to name the folders
    randPatValue = R.randint(1000000000000, 9999999999999)
    # Patient folder where data files are originally contained
    patFolderPath = allPatFolder + '/' + patient
    # Define patient output folder
    patOutFolderPath = outFolder + '/' + str(randPatValue)
    # Copy the data
    # shutil.copy(patFolderPath, patOutFolderPath) 
    copy_tree(patFolderPath, patOutFolderPath)
    # Print conversion table
    print(patient + ' ' + str(randPatValue))

print('Program is complete!')

