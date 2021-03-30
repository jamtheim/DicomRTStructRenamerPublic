# This script will QA the new data.
# Checks if multiple struct files exist or if multiple CT series are present
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


def getAllStructureFiles(path):
    """
    Search a given path for all structure DICOM files
    Inputs:
        path (str): Path to the DICOM directory
    Returns:
        Number of RT file names existning in the directory
    """
    files = os.listdir(path)
    structFiles = [f for f in files if ".dcm" in f]
    structFiles = [f for f in structFiles if "RS." in f]
    nrRTfiles = len(structFiles)
    if len(structFiles) == 0:
        nrRTfiles = 0
    return nrRTfiles


def getScanInfo(patCTFolder):
    # Get the files
    files = os.listdir(patCTFolder)
    # Make sure they are dicom
    files = [f for f in files if ".dcm" in f]
    # Make sure it is a CT file
    CTfiles = [f for f in files if "CT" in f]
    # Define number of CT files
    nrCTFiles = len(CTfiles)
    # print(scanDate)
    return nrCTFiles


def getExpectedNrCTFiles(patCTFolder):
    # Get all the file names
    files = os.listdir(patCTFolder)
    # Make sure it is only CT file collected
    files = [f for f in files if "CT" in f]
    # Sort the files
    files.sort()
    # Remove dcm ending from file name string
    files = [f.replace(".dcm", "") for f in files]
    # Get the last four digits of the first file in the array and trip away zeroes from left
    firstFileNr = files[0][-4:].lstrip('0')
    # Get the last four digits of the last file in the array and trip away zeroes from left
    lastFileNr = files[-1][-4:].lstrip('0')
    # Calculate numeric difference
    expectedNRCTFiles = int(lastFileNr) - int(firstFileNr)
    # Add one for correctness
    expectedNRCTFiles = expectedNRCTFiles + 1
    # This method does not account for the fact that the first or last slice might be missing
    # Take care of checking first slice. Do not know how to check last slice. 
    firstSliceDicom = pydicom.dcmread(patCTFolder + '/' + files[0] + '.dcm')
    # Check that it has instance number 1
    # if condition returns False, AssertionError is raised
    assert firstSliceDicom.InstanceNumber==1
    return expectedNRCTFiles


def checkSeriesInstanceUID(patient):
    # Define patient CT folder
    patCTFolder = allPatFolder + '/' + patient  
    # Define files
    files = os.listdir(patCTFolder)
    # Get only DICOM files 
    files = [f for f in files if ".dcm" in f]
    # Make sure it is a CT file
    CTfiles = [f for f in files if "CT" in f]

    # Define seriesInstanceUID ground truth to compare with
    seriesInstanceUID_GT= pydicom.dcmread(patCTFolder + '/' + CTfiles[0]).SeriesInstanceUID
    # Check every file in the patient folder if series instance UID is correct. 
    for ctfile in CTfiles: 
        # Get the series instance UID for each file and compare to GT
        seriesInstanceUID= pydicom.dcmread(patCTFolder + '/' + ctfile).SeriesInstanceUID
        assert seriesInstanceUID == seriesInstanceUID_GT
        print('Dicom file check for series instance UID')  

# Define what dataset to process
procDataSet = 'NewData'

# For new large dataset
if procDataSet == 'NewData': 
    # Patient folder
    allPatFolder = '/mnt/mdstore2/Backup/Christian/BigDataForStructFinder/ProstateDataBigRaw/'
    # List patients in patFolder
    patFolders = os.listdir(allPatFolder)

    def patLoop(patNr, patient):
        # Define patient CT folder
        patCTFolder = allPatFolder + '/' + patient  
        # Get nr of CT slices
        nrCTfiles = getScanInfo(patCTFolder) 
        # Get number of RT struct files
        nrRTStructFiles = getAllStructureFiles(allPatFolder + '/' + patient)
        # Get number of expected CT files (from file naming)
        nrExpectedCTFiles = getExpectedNrCTFiles(patCTFolder) 
        # Return data
        return nrCTfiles, nrRTStructFiles, nrExpectedCTFiles

    # Init lists and counter
    nrCTfilesAllPatients = []
    nrRTfilesAllPatients = []
    nrExpectedCTFilesAllPatients = []
    countDeviationRTstruct = 0

    # Loop over all patients
    for patNr, patient in enumerate(patFolders): 
        print(patient)
        # Print status for every 100 patient
        if (patNr % 100) == 0:
            print('Processing patient nr ' + str(patNr))
        # Get number of available CT and RT struct files for each patient and collect them 
        # Also get expected number of CT files (from the CT file naming)
        nrCTfiles, nrRTStructFiles, nrExpectedCTFiles = patLoop(patNr, patient)   
        # Collect
        nrCTfilesAllPatients.append(nrCTfiles)
        nrRTfilesAllPatients.append(nrRTStructFiles)
        nrExpectedCTFilesAllPatients.append(nrExpectedCTFiles)

        # Check if number of available CT files is the same as expected number of files 
        if nrCTfiles != nrExpectedCTFiles: 
            print(patient)
            print('Available ' + str(nrCTfiles) + ' files but expected ' + str(nrExpectedCTFiles) + ' files')
            print('Deviation in expected and available number of CT files')


        # For patients with more than one RT struct file
        if nrRTStructFiles > 1:
            # checkSeriesInstanceUID(patient)
            print(patient)
            print(nrRTStructFiles)
            print(nrCTfiles)
            countDeviationRTstruct += 1
            print(' ')
        # If patient only has one RT struct make sure all 
        # CT files belong to the same series. 
        # elif nrRTStructFiles == 1: 
            # checkSeriesInstanceUID(patient)
            # print('Patient checked')  
        elif nrRTStructFiles == 0: 
            raise Exception('Patient seem to be missing RT struct file')
        
    # Summary information
    print(str(countDeviationRTstruct) + ' patients has more than one RT struct' )

    #nrCPU = multiprocessing.cpu_count()
    #nrCPU = 1
    # Init parallell job
    #Parallel(n_jobs=nrCPU, verbose=10)(delayed(patLoop)(patNr, patient) for patNr, patient in enumerate(patFolders))
print('Program has completed! ')
