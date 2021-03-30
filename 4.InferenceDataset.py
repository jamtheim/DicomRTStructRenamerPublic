# File for running inferens of the model on a patient dataset.
# Dataset needs a specific file structure where dicom files and RT struct has its place
# Sanity check is performed for the determined label. 
# Output of saliency map can be enabled
# Author: Christian Jamtheim Gustafsson, PhD

import os
import cv2
import csv
import numpy as np
import os.path
import pydicom
import random
import scipy
import time
from glob import glob
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dcmrtstruct2nii import dcmrtstruct2nii, list_rt_structs
import shutil
import operator
from commonConfig import loadModel, loadModelInputConf, loadStructuresOfInterest, loadIgnoreStructures
from myPreProcessingClass import myPreProcessing
from myInferenceClasses import myInferencePrepare
from sharedFunctions import getStructureFile, RemoveItemsFromList
from joblib import Parallel, delayed
import multiprocessing
from myExplainClasses import myExplainModels

# Init myPreProcess class instance
PreProcess = myPreProcessing()
# Init myInference class instance
myInference = myInferencePrepare()
# Init myExplain class instance
myExplain = myExplainModels()

# Global Settings
organ = 'Prostate'
# Use model
useModel = 'InceptionResNetV2'
# Produce explainable maps
createExplainMaps = 0
# Create list of structures 
# This is only to resolve the new name after inference
InferenceStructureNames  =  loadStructuresOfInterest(organ,'inference') 
# Paths
ProjectPath = '/mnt/mdstore1/Christian/Projects/StructFinder'
datasetFolder = ProjectPath + '/' + 'dataset' + '/' + 'Inference'
# Determine what dataset to do inference on
# infDataSet = 'MRProtect_MRI'
# infDataSet = 'ProstateDataBigRawSortedTestDatasetAnon_pat1-50'
# infDataSet = 'ProstateDataBigRawSortedTestDatasetAnon_pat51-200'
# infDataSet = 'MRProtect_CT'
# infDataSet = 'NotSeenTest'
# infDataSet = 'Explainability'
infDataSet = 'Umea'

# Define name of that specific datafolder
dataFolder = datasetFolder + '/' + infDataSet
# Define CT/MRI and RTStructure folder names
CTFolderName = 'CT'
RTStructFolderName = 'RTStruct'
EditedRTStructPrefix = 'edited_'
# Determine location for model folders
modelFolderBase = 'models'
# Determine what model experiment iteration to use
expIter = 109
# Define how many cross validations were used for that experiment iteration
# This determines number of models in majority voting
nrCrossVal = 10
# Determine what saved epoch number of the models to use
modelEpochIter = 100
# Determine how many models that must agree before raising a flag
freqModelsAgreeingFlagLevel = 6
# Determine at what threshold the mean majority probability flag should be raised
meanMajModelProbThresh = 0.7
# Defined percentiles for volume QC. Volume outisde this range will be flagged. 
lowPercentileVolumeQC = 1
highPercentileVolumeQC = 99

# Define experiment model epochs save folder
modelExpFolder = ProjectPath + '/' + modelFolderBase + \
    '/' + 'StructFinder_exp' + str(expIter)

# Model configuration
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# For using CPU, do this:
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Load model input configuration
modelInputConfig = loadModelInputConf(organ)
# Get config
useChannels = modelInputConfig.useChannels
useClasses = modelInputConfig.useClasses
useResolution = modelInputConfig.useResolution
# Load all selected models from the cross validations
AllModels = myInference.loadAllCVModels(modelExpFolder, nrCrossVal, modelEpochIter, organ, useModel)

# Config for QC - Sanity check using the label/volume dependence
# Load database from training data
databaseVersion = 42
# Dataset folder
trainingDatasetFolder = ProjectPath + '/' + 'dataset/Final' + '/' + 'StructFinderDataset' + organ + 'V' + str(databaseVersion)
# Volume Label database file
objectVolumeDatabaseFilePath = trainingDatasetFolder + '/' + 'ObjectVolumeDatabase.csv'
# Get volume statistics for all classes
meanVolumeAllClasses, stdVolumeAllClasses, minVolumeAllClasses, maxVolumeAllClasses, dataVolumeAllClasses = myInference.getVolumeLabelStatistics(objectVolumeDatabaseFilePath, useClasses)

# Get list of patients in dataFolder
patFolders = os.listdir(dataFolder)
# Set counter
counter = 0
# For each patient in the dataset
# Original loop
for patient in patFolders:
    # Start of patient loop
    counter = counter+1
    # Define CT folder
    patCTFolder = dataFolder + '/' + patient + '/' + CTFolderName
    # Define temporary output folder with random number as suffix
    tempOutFolder = datasetFolder + '/' + infDataSet + '_tmpOut' + '/' + patient + '_' + str(random.randrange(1000000000))
    tempOutInfDataFolder = tempOutFolder + '/infData'
    # Create folders
    if not os.path.exists(tempOutFolder):
        os.makedirs(tempOutFolder)
    if not os.path.exists(tempOutInfDataFolder):
        os.makedirs(tempOutInfDataFolder)
    # Get RT struct file
    RTStructFile = getStructureFile(
        dataFolder + '/' + patient + '/' + RTStructFolderName)
    # Define path for RT struct file
    patRTStructFile = dataFolder + '/' + patient + \
        '/' + RTStructFolderName + '/' + RTStructFile
    # Define path for new/edited RT struct file
    editedPatRTStructFile = dataFolder + '/' + patient + '/' + EditedRTStructPrefix + \
        RTStructFolderName + '/' + EditedRTStructPrefix + RTStructFile
    # Create folder for edited RT struct
    if not os.path.exists(dataFolder + '/' + patient + '/' + EditedRTStructPrefix + RTStructFolderName):
        os.mkdir(dataFolder + '/' + patient + '/' +
                 EditedRTStructPrefix + RTStructFolderName)
    # Define file to write results to 
    outFileNameResults = dataFolder + '/' + patient + '/' + patient + '_resultsInference.csv'
    # Define file to write errors to 
    outFileNameErrors = dataFolder + '/' + patient + '/' + patient + '_errorsInference.txt'
    # Define file object (appendable writing)
    outFileObjectResults = open(outFileNameResults, "a")
    # Write header information for results
    outFileObjectResults.write('Predicted structure name' + '\t' + 'Given structure name' + '\t' + 'Predicted by # models' + '\t' + 'Mean majority probability' + '\t' + 'Agreeing models' + '\t' + 'QC fail' + '\t' + 'QC fail reason')
    # Insert new line
    outFileObjectResults.write("\n")
    # Close file write
    outFileObjectResults.close()

    # Print patient and counter
    print(patient)
    print(counter)
    # Convert the RT structs to Nifty format        
    print('Convert RT structures to Nifti files')
    # Get list of all structures present
    structListAuto = list_rt_structs(patRTStructFile)
    # Load list of structures to ignore (help structures, X,Y, Z...)
    ignoreStructures = loadIgnoreStructures(organ)
    # Remove items in that list according to the selection to ignore
    structListCleaned, existFlag = RemoveItemsFromList(structListAuto, ignoreStructures)
    
    # Extract the structures that are left in parallell with a special function 
    # If fail, it will not interupt the other structure extractions from the RT struct set.   
    def dcmrtstruct2niiLoop(currStruct): 
        try: 
            # Extract the structure and convert to Nifti
            # We do not want convert_original_dicom=True for all structures as this will add a lot of time. 
            # Do this only for BODY as this structure is always present. It has nothing to do with the structure itself for enabling convert_original_dicom=True. 
            if currStruct == 'BODY' or currStruct == 'External':
                dcmrtstruct2nii(patRTStructFile, patCTFolder, tempOutFolder, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=True)
            else: 
                dcmrtstruct2nii(patRTStructFile, patCTFolder, tempOutFolder, structures=currStruct, gzip=True, mask_background_value=0, mask_foreground_value=1, convert_original_dicom=False)

        except: 
            print('An exception occurred in dcmrtstruct2nii data extraction for struct ' + str(currStruct))
            print(patRTStructFile)
            print(list_rt_structs(patRTStructFile))
            # Write to error log file,
            myInference.write2log(outFileNameErrors, 'An exception occurred in dcmrtstruct2nii data extraction for struct ' + str(currStruct))
        
    # Count number of CPUs and assign whats needed
    nrCPU = multiprocessing.cpu_count()
    if len(structListCleaned) < nrCPU: 
        nrCPU = len(structListCleaned)
    # Init parallell job
    Parallel(n_jobs=nrCPU, verbose=10)(delayed(dcmrtstruct2niiLoop)(currStruct) for structNr, currStruct in enumerate(structListCleaned))                  

    # Load the RT struct DICOM file and assign (copy) data to new variable
    RTStructure = pydicom.dcmread(patRTStructFile)
    RTStructureEdited = RTStructure
    # Get RT struct nii.gz file names that have been outputed above from the cleaned list using dcmrtstruct2nii
    structListFilesCleaned = os.listdir(tempOutFolder)
    # Make sure the list only contains nii.gz files
    structListFilesCleaned = [file for file in structListFilesCleaned if '.nii.gz' in file]
    # Make sure the list only contains files with prefix 'mask_'. The image data (image.nii.gz) of the CT files is then excluded. 
    structListFilesCleaned = [file for file in structListFilesCleaned if 'mask_' in file]

    # Create AllStructsArray and body struct (parallelized)
    AllStructsArray, bodyStruct = myInference.CreateStructureFusion(
        structListFilesCleaned, tempOutFolder, patient)
    # This is the section of interest to automatically edit the ROI names in the RT struct file.
    # Info regarding RT Struct DICOM, ROI name ftp://medical.nema.org/medical/dicom/final/sup11_ft.pdf
    # Loop through all structures in the DICOM structure.    
    for index, currentROISequence in enumerate(RTStructure.StructureSetROISequence):
       # Get ROI name using index
        ROINameCollected = RTStructure.StructureSetROISequence[index].ROIName
        # This name should be the same as the one that dcmrtstruct2nii reads from its uncleaned list
        # Nice to know that they are the same
        if structListAuto[index] != ROINameCollected:
            print('BeAware: RT structure names are not the same between DICOM file and dcmrtstruct2nii cleaned list')
            # Write to error log file
            myInference.write2log(outFileNameErrors, 'BeAware: RT structure names are not the same between DICOM file and dcmrtstruct2nii cleaned list')

        # If ignoreStructures is not included in the the ROINameCollected structure execute further actions
        ROINameCollectedCheck = [ROINameCollected] # Important brackets for it to work
        trashDataIgnore, existInIgnoreFlag = RemoveItemsFromList(ROINameCollectedCheck, ignoreStructures)      
        if existInIgnoreFlag == 0: 
            # Structure names in list_rt_structs does not correspond to outputed filenames for the nii.gz files
            # as some characters are removed and some character changed to something else in the output of dcmrtstruct2nii.
            # The function below adjusts this.
            ROINameForMaskFile = myInference.adaptROINameForMaskFile(ROINameCollected)
            # Try to access the outputed nii.gz file using the adapted name ROINameForMaskFile
            # I want to make sure there is a valid connection between the DICOM ROI name and the generated nii.gz file.
            fileNameDetermined = 'mask_' + ROINameForMaskFile + '.nii.gz'
            print(ROINameCollected)
            
            # Create needed inference data for the current structure. This writes the 2D projection and BodyAndOtherAddMap slice.
            # Output a bool imgDataExist, reflecting non empty image data in fileNameDetermined. 
            # This outputs file name paths also, needed for loading data into the model.
            filePathOut2DTraProj, filePathOut2DCorProj, filePathOut2DSagProj, filePathOutBodyAndOtherAdd2D, imgDataExist = myInference.CreateInferenceData(
                AllStructsArray, bodyStruct, fileNameDetermined, tempOutFolder, tempOutInfDataFolder)
        
            if imgDataExist == 1:
                # Get volume data for the current file
                # Read file
                currentNiiImage,currentStructData = PreProcess.nii2nparray(tempOutFolder + '/' + fileNameDetermined)
                # Get voxel size of the current image as 3 dim vector and QA it
                voxelSize = currentNiiImage.header.get_zooms()
                assert len(voxelSize) == 3
                # Get the volume of the current 3D structure
                # As we are dealing with binary mask images, we can sum the signal of 1 valued voxels
                # and then multiply with voxel size, defined in mm 
                currentStructVolume = round(np.sum(currentStructData[:]) * voxelSize[0] * voxelSize[1] * voxelSize[2])

                # If image data exist send data to network for inference
                bz = 1 #batch size
                # Init matrix 
                X = np.zeros((bz, useResolution[0], useResolution[1], useChannels)) # Use zero to enable QA check below
                # Add image information into channels
                # Loading this from nii file for it to be exactly the same as in the training (preprocessing wise) 
                # Nii files are created again in BodyAndOtherAddMap, so output from dcmrtstruct2nii is not used here.
                # Get only first argument from function which is the numpy data
                X[bz-1, :, :, 0] = PreProcess.loadAndResize(filePathOut2DTraProj, useResolution[0], useResolution[1])[0]
                X[bz-1, :, :, 1] = PreProcess.loadAndResize(filePathOutBodyAndOtherAdd2D, useResolution[0], useResolution[1])[0]
                X[bz-1, :, :, 2] = PreProcess.loadAndResize(filePathOut2DCorProj, useResolution[0], useResolution[1])[0]
                X[bz-1, :, :, 3] = PreProcess.loadAndResize(filePathOut2DSagProj, useResolution[0], useResolution[1])[0]

                # QA check to make sure non zero data was loaded into each channel before inference
                for i in range(0,X.shape[3]):
                    assert np.sum(X[:, :, :, i][:]) != 0

                # Majority voting and inference prediction
                mostCommonIndex,freqModelsAgreeing,collProbMajorityAll,modelsAgreeing = myInference.predictAndVote(X, AllModels)    
                # If no majority vote could be determined
                if mostCommonIndex == float("NaN"):
                    myInference.write2log(outFileNameErrors, 'Majority vote could not be determined') 
                    raise ValueError('Majority vote could not be determined')
                else:
                    # QC check depending on volume interval for the label, number of agreeing models or mean majority model probability. 
                    # Value = True = QC OK, False = QC not OK (fail). 
                    # Old method: QCStatus = myInference.checkQCVolumeInferenceNormalDist(currentStructVolume, mostCommonIndex, meanVolumeAllClasses, stdVolumeAllClasses, 2)
                    QCVolumeStatus = myInference.checkQCVolumeInferencePercentile(currentStructVolume, mostCommonIndex, dataVolumeAllClasses, lowPercentileVolumeQC, highPercentileVolumeQC)
                    if QCVolumeStatus == False:
                        QCStatusPrint = '1'
                        QCFailReason = 'Volume'
                        print('QC have detected the volume ' + str(fileNameDetermined) + ' to be a volume outlier')
                        myInference.write2log(outFileNameErrors, 'QC have detected the volume ' + str(fileNameDetermined) + ' to be a volume outlier')
                    # Check model agreement
                    elif freqModelsAgreeing < freqModelsAgreeingFlagLevel: 
                        QCStatusPrint = '1'
                        QCFailReason = 'MajVote'
                        print('QC have detected a deviation in model agreement for ' + str(fileNameDetermined))
                        myInference.write2log(outFileNameErrors, 'QC have detected a deviation in model agreement for ' + str(fileNameDetermined))
                    # Check majority model mean probability 
                    elif np.mean(collProbMajorityAll) < meanMajModelProbThresh: 
                        QCStatusPrint = '1'
                        QCFailReason = 'MajProb'
                        print('QC have detected a low mean model majority probability for ' + str(fileNameDetermined))
                        myInference.write2log(outFileNameErrors, 'QC have detected a low mean model majority probability for ' + str(fileNameDetermined))
                    else: 
                        QCStatusPrint = '0'
                        QCFailReason = 'N/A'
                
                    # Get new name and print it 
                    predictedROIName = InferenceStructureNames[mostCommonIndex]
                    print(predictedROIName + ' is the predicted name selected from ' + str(freqModelsAgreeing) + '/' + str(len(AllModels)) + ' models with a mean majority probability of ' + str(np.round(np.mean(collProbMajorityAll)*100)) + ' %')
                    print('Following models agree: ' + str(modelsAgreeing))
                                        
                    # Define file object (appendable writing)
                    outFileObjectResults = open(outFileNameResults, "a")
                    # For every name in the list, write out data and QC status on a new line in the result file
                    outFileObjectResults.write(predictedROIName + '\t' + ROINameCollected + '\t' + str(freqModelsAgreeing) + '\t' + str(np.mean(collProbMajorityAll)) + '\t' + str(modelsAgreeing).replace(',', '') + '\t' + str(QCStatusPrint) + '\t' + str(QCFailReason) )
                    # Insert new line
                    outFileObjectResults.write("\n")
                    # Close file write
                    outFileObjectResults.close()

                    # This section adds dose information to the structure name if the structure inference name contains GTV, CTV or PTV. 
                    # Added information is only added to the DICOM RT structure. In the result file above I have not included it. 
                    # Use ROINameCollected to extract dose information and not nii2struct mask names as it has discarded the decimal point for dose. 
                    # Check if it is a target according to inference name
                    isTarget = myInference.checkIfTarget(predictedROIName)
                    # If so, get dose from ROINameCollected else set to empty
                    if isTarget:
                        doseLevel = myInference.getTargetDose(ROINameCollected)
                        doseLevelAddon = '_' + doseLevel
                    else: # If not a target
                        doseLevelAddon = ''

                    # Insert new ROI name in the StructureSetROISequence[index].ROI name with added dose level
                    RTStructureEdited.StructureSetROISequence[index].ROIName = predictedROIName + doseLevelAddon
                    print(predictedROIName + doseLevelAddon)
                    
                    # Create explainability maps if enabled
                    if createExplainMaps: 
                        myExplain.CreateExplainMaps(patient, X, ROINameCollected, AllModels, mostCommonIndex)

            else:
                # Do nothing, output warning
                print('Data does not exist for the structure ' + str(fileNameDetermined))
                myInference.write2log(outFileNameErrors, 'Data does not exist for the structure ' + str(fileNameDetermined))
        else: 
            print(str(ROINameCollected) + ' was ignored in inference and will not be edited') 
            myInference.write2log(outFileNameErrors, str(ROINameCollected) + ' was ignored in inference and will not be edited')

    # Save RT structure DICOM file with new names edited 
    pydicom.dcmwrite(editedPatRTStructFile, RTStructureEdited,
                     write_like_original=True)
    # End of patient loop

print('Program has completed')
