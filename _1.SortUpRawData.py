# This script will sort up the DICOM raw data into sub folders for CT and RTStructure
# Author: Christian Jamtheim Gustafsson, PhD

import os
import cv2
import numpy as np
import os.path
import pydicom
import scipy
from random import seed
from random import randint
import shutil
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
from sharedFunctions import moveRTStructFile, moveCTFiles
import shutil


allPatFolder = '/mnt/mdstore2/Backup/Christian/BigDataForStructFinder/ProstateDataBigRawSorted/'
# allPatFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/Debug/'
# List patients in patFolder
patFolders = os.listdir(allPatFolder)
# Loop over all patients 
for patNr, patient in enumerate(patFolders): 
    # Patient folder where data files are originally contained
    patFolderPath = allPatFolder + '/' + patient
    # Move RS struct file that is not a CT Dicom file 
    RTStructFolderName = moveRTStructFile(patFolderPath)
    # Move CT files to a subdirectory under the patient
    CTFolderName = moveCTFiles(patFolderPath)
    # Make sure there are only 2 items (folders) left under each patient
    # That means all the files have been moved to their corresponding folder
    patFolderItems = os.listdir(patFolderPath)
    assert len(patFolderItems) == 2

print('Program is complete!')
