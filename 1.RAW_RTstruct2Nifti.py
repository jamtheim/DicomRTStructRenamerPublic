# This script will generate Nifti files from the RT structures and also output the CT series in Nifti format. 
# Author: Christian Jamtheim Gustafsson, PhD

import os
import cv2
import numpy as np
import os.path
import pydicom
import scipy
import nibabel as nib
from nibabel.testing import data_path
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import matplotlib.pyplot as plt
from random import seed
from random import randint
import shutil
import pydicom
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import multiprocessing
from sharedFunctions import * 


# Define what dataset to process
procDataSet = 'ProstateDataBigRawSortedNoTestData'

# For the large prostate patient dataset
if procDataSet == 'ProstateDataBigRawSortedNoTestData': 
    # Patient folder
    allPatFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/' + procDataSet + '/'
    #allPatFolder = '/mnt/mdstore2/Backup/Christian/BigDataForStructFinder/NewDataSorted/'
    # List patients in patFolder
    patFolders = os.listdir(allPatFolder)
    # Define output folder for Nifty data
    outFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/' + procDataSet + '_Nifti'
    #outFolder = '/mnt/mdstore2/Backup/Christian/BigDataForStructFinder/NewNifty/'

    # Make sure the output directory exists
    if not os.path.isdir(outFolder):
        # Create dir
        os.mkdir(outFolder)


    def patLargeDataLoop(patNr, patient):
        # Create new random seed
        # Important for parallell threading
        R=np.random.RandomState()
        # Create random large integer
        # Use it to name the folders
        randPatValue = R.randint(100000000000, 999999999999)
        # Patient folder where data files are originally contained
        patFolderPath = allPatFolder + '/' + patient
        # Move RS struct file that is not a CT Dicom file to another folder
        # otherwise dcmstruct2nii will not accept it
        RTStructFolderName = moveRTStructFile(patFolderPath)
        # Move CT files to a subdirectory under the patient
        CTFolderName = moveCTFiles(patFolderPath)
        # Patient folder where CT files now are contained
        patCTFolderPath = allPatFolder + '/' + patient +'/' + CTFolderName
        # Get RT struct file name
        RTStructFile = getRTStructFile(patFolderPath + '/' + RTStructFolderName)
        # Define whole path for RT struct file
        patRTStructFile = patFolderPath + '/' + RTStructFolderName + '/' + RTStructFile
        # Define patient output folder
        patOutFolderPath = outFolder + '/' + str(randPatValue)
        # Get list of all structures present
        structListExported = list_rt_structs(patRTStructFile)

        # Convert the RT structs to Nifty format 
        # This is performed by targeting each individual structure at a time in a loop. 
        # This is slower but safer. 
        # In this way we can isolate exceptions to individual structures and not break 
        # the process of dcmrtstruc2nii which happens otherwise. This avoids modification of the 
        # dcmrtstruc2nii source code and allows us in retrospect to see if missing data was important or not.
        # Failed objects are due to the fact that the structures are not completed or simply empty in Eclipse. 
        for structNr, currStruct in enumerate(structListExported):
            try:
                # Extract the structure and convert to Nifty
                # We do not want convert_original_dicom=True for all structures as this will add a lot of time. 
                # Do this only for BODY as this structure is always present. It has nothing to do with the structure itself for enabling convert_original_dicom=True. 
                if currStruct == 'BODY':
                    print(patient   + '  ' + str(randPatValue))
                    dcmrtstruct2nii(patRTStructFile, patCTFolderPath, patOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=True)
                else:
                    dcmrtstruct2nii(patRTStructFile, patCTFolderPath, patOutFolderPath, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=False)
                
            except:
                print("Exception when extracting " + currStruct + ' for ' + patient  + ' ' + str(randPatValue))


    nrCPU = multiprocessing.cpu_count()
    # nrCPU = 1
    # Init parallell job
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(patLargeDataLoop)(patNr, patient) for patNr, patient in enumerate(patFolders))


print('Program is complete!')

