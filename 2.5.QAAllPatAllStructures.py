# File for outputting PNG images from all structures from all patients 
# This can be used as a QA tool for the data
# Please observe that changes which are made to the labels in the label correction file will not 
# show its effect in the PNG images produced here. 
# Author: Christian Jamtheim Gustafsson, PhD

import os
import cv2
import numpy as np
import os.path
import pydicom
import scipy
from glob import glob
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import SimpleITK as sitk
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
from sharedFunctions import * 
from commonConfig import loadStructuresOfInterest, loadModelInputConf
from joblib import Parallel, delayed
import multiprocessing
from myPreProcessingClass import myPreProcessing
# Init PreProcess class instance
PreProcess = myPreProcessing()

# Define functions
def centerSlicesExtract(structureName, folderOfInterest, patFolders): 
    """
    Collects the ROIs (mask name) of interest and output pngs for the middle slice of the tra and cor slice 
    Do this for all patients in the patient folder 
    Inputs:
        ROI name 
    Returns:
        Creates PNG files for the middle slices
    """
    # For each patient 
    for patNr, patient in enumerate(patFolders):
        # Get mask and file name
        maskFileName, filenamePath = getFileNamePathForStructure(structureName, specialStructs, folderOfInterest, patient)
        # Check if the filename exists for the specific structure and create PNG for the middle transversal and coronal slice
        if os.path.isfile(filenamePath) == 1: 

            ## OLD METHOD. I CHANGED THIS TO USE THE SAME AS OTHER PLACES IN MY SOFTWARE. 
            # I DID THIS TO EXCLUDE ANY RISK OF VIEWING THE WRONG DATA IN QA COMPARED TO WHATS USED
            # Read the 3D image with Simple ITK
            #sitk_Image = sitk.ReadImage(filenamePath)
            # Other data in the project is loaded with 
            # currentNiiStruct,currentStructData = PreProcess.nii2int8data(filenamePath)
            # This uses nibabel instead. Nibabel does however not get the image
            # orientation right for the output but Simple ITK does. 
            # Output of PNG will then be rotated and LR flipped.
            # It does not matter how nibabel treats data in the learning process as long as the output is consistant. 
            # Access the numpy array
            # itk_image = sitk.GetArrayFromImage(sitk_Image)
            # Reorder axis so slice axis is in third place 
            # currentStructData = np.transpose(itk_image, (1, 2, 0))
            ## OLD METHOD END

            # Read data 
            currentNiiStruct,currentStructData = PreProcess.nii2nparray(filenamePath)
            # Get the center of mass slices for the currentStructData
            centerRowSlice,centerColSlice,centerTraSlice=getBinaryArrayCoM(currentStructData)
            # Define data for center tra slice
            writeTraData = currentStructData[:,:,centerTraSlice]
            # Define data for center cor slice
            writeCorData = currentStructData[:,centerColSlice,:]

            # New method - projected data
            writeTraProjData = getBinaryArray2DProjection(currentStructData,'tra','inDataRotatedTrue')
            writeCorProjData = getBinaryArray2DProjection(currentStructData,'cor','inDataRotatedTrue')

            # Check tra slice if empty
            if np.sum(writeTraData[:]) == 0: 
                print(filenamePath + ' has empty TRA data')
            if np.sum(writeTraProjData[:]) == 0: 
                print(filenamePath + ' has empty TRA proj data')

            # Check cor slice if empty
            if np.sum(writeCorData[:]) == 0: 
                print(filenamePath + ' has empty COR data')
            if np.sum(writeCorProjData[:]) == 0: 
                print(filenamePath + ' has empty COR proj data')


            # Make sure directory exists
            if not os.path.exists(dataSetFolderOut + '/' + structureName):    
                # Create directories
                os.makedirs(dataSetFolderOut + '/' + structureName)
            # Save data to PNG
            # Output of PNG will be rotated and LR flipped for TRA. I correct for this bellow. 
            # Output of PNG will be rotate for COR. I correct for this bellow. 
            plt.imsave(dataSetFolderOut + '/' + structureName + '/' + str(patient) + '_' + maskFileName + '_tra.png', cv2.flip(cv2.rotate(writeTraData, cv2.ROTATE_90_CLOCKWISE),1))
            plt.imsave(dataSetFolderOut + '/' + structureName + '/' + str(patient) + '_' + maskFileName + '_cor.png', cv2.rotate(writeCorData, cv2.ROTATE_90_COUNTERCLOCKWISE))
            # New data
            plt.imsave(dataSetFolderOut + '/' + structureName + '/' + str(patient) + '_' + maskFileName + '_traProj.png', cv2.flip(cv2.rotate(writeTraProjData, cv2.ROTATE_90_CLOCKWISE),1))
            plt.imsave(dataSetFolderOut + '/' + structureName + '/' + str(patient) + '_' + maskFileName + '_corProj.png', cv2.rotate(writeCorProjData, cv2.ROTATE_90_COUNTERCLOCKWISE))
            
            # Display slice
            # imgplot = plt.imshow(writeData)
            # plt.show(block=False)
            # plt.pause(0.2)
            # plt.close()
            
            

# Main Code Here
ProjectPath = '/mnt/mdstore1/Christian/Projects/StructFinder'
organ = 'Prostate'
version = 31
RawDataInput = ProjectPath + '/' + 'dataset/RAW' + '/' + 'ProstateDataBigRawSortedNoTestData_Nifti/'
# RawDataInput = '/mnt/mdstore1/Temp/HypoOutHypoTest'
dataSetFolderOut = ProjectPath + '/' + 'dataset/QA' + '/' + organ + 'V' + str(version)
# Define folders of interest
folderOfInterest = RawDataInput
folderOutput = dataSetFolderOut
# Create folders
if not os.path.exists(folderOutput):
    os.makedirs(folderOutput)

# Get all patient folders
patFolders = os.listdir(folderOfInterest)
# Get config to use
modelInputConfig = loadModelInputConf(organ)
useResolution = modelInputConfig.useResolution
# Create list of structures of interest to loop over
# This list is created in lower letters
AllStructuresOfInterest, structs, specialStructs, AllLabelsOfInterest =  loadStructuresOfInterest(organ,'training') 

def patientLoop(currStructure): 
    # print(currStructure)
    # Execute 
    centerSlicesExtract(currStructure, folderOfInterest, patFolders)

# Count number of CPUs
nrCPU = multiprocessing.cpu_count()
# nrCPU = nrCPU
# Init parallell job
Parallel(n_jobs=nrCPU, verbose=10)(delayed(patientLoop)(currStructure) for structNR, currStructure in enumerate(AllStructuresOfInterest))