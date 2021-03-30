# File for sorting all patient Nifti files to folders consisting of organ name
# Also creates linear combination images (organs and BODY) and label files
# Also create HDF5 database for images, labels and volumes
# Author: Christian Jamtheim Gustafsson, PhD

import os
import cv2
import numpy as np
import os.path
import pydicom
import pandas as pd
import scipy
import h5py
import csv
from glob import glob
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import asarray
import shutil 
from joblib import Parallel, delayed
import multiprocessing
from sharedFunctions import (saveNpSlice2nii, getFileNamePathForStructure, QACheckMatrixSize, RemoveItemsFromList, 
fuseStructures, getBinaryArray2DProjection, BodyAndOtherAddMap, saveArray2nii2DCor, saveArray2nii2DTra, saveArray2nii2DSag, saveArray2nii3D, getBinaryArrayCoM)
from myPreProcessingClass import myPreProcessing
from commonConfig import loadStructuresOfInterest, loadIgnoreStructures, getSignalTruncTh, loadModelInputConf

# Init PreProcess class instance
PreProcess = myPreProcessing()

def checkNotEmpty(npStructData, sliceOfInterest): 
    """
    Check if sliceof interest is empty or not
    """
    if np.sum(npStructData[:,:,sliceOfInterest][:]) == 0: 
        return False 
    else: 
        return True


def collectAllStructsAvailable(ignoreStructures, folderOfInterest, patient): 
    """
    Collect all available structures of the patient and sum them up in one array
    This array does not include the BODY mask. BODY mask is outputed stand alone. 
    Exclude structures from a list given in the input
    Inputs:
        ignoreStructures, folderOfInterest, patient
    Returns:
    AllStructsArray, dataBody
    """
    # Get image volume size from the BODY structure, which always exist in each patient
    maskFileNameBody = 'mask_' + 'BODY'
    # Get the full path of the BODY file
    filenamePathBody = folderOfInterest + '/' + patient + '/' + maskFileNameBody + '.nii.gz'
    # Get nii and body array data from nii file
    niiBody,dataBody = PreProcess.nii2nparray(filenamePathBody)
    # QA Check for matrix size
    QACheckMatrixSize(dataBody)
    # Assign zeros to an empty array
    AllStructsArray = np.zeros(dataBody.shape) 
    # Get all structure files available 
    availableStructureFiles = os.listdir(folderOfInterest + '/' + patient)
    # Make sure the list only contains nii.gz files
    availableStructureFiles = [file for file in availableStructureFiles if '.nii.gz' in file]   
    # Remove CT image file 
    availableStructureFiles = [file for file in availableStructureFiles if 'image.nii.gz' not in file]   
    # Remove BODY mask 
    availableStructureFiles = [file for file in availableStructureFiles if maskFileNameBody not in file]  
    # Remove file endings and get avilable structure list
    availableStructures = [file.replace('.nii.gz','') for file in availableStructureFiles if '.nii.gz' in file]  
    # Remove mask_ prefix
    availableStructures = [file.split('mask_')[1] for file in availableStructures if 'mask_' in file]
    # Remove structures to ignore 
    availableStructuresCleaned, existFlag = RemoveItemsFromList(availableStructures, ignoreStructures)
    # Collect all data from the structures in the final list 
    # This is not parallellized because we parallelize calculation over patients. 
    for structNrOther, currStructure in enumerate(availableStructuresCleaned):
        currFilePath = folderOfInterest + '/' + patient + '/' + 'mask_' + currStructure + '.nii.gz' 
        AllStructsArray= fuseStructures(AllStructsArray, currFilePath)

    return AllStructsArray, dataBody


def loadCorrectionStructureLabels(correctionLabelsCSV): 
    """
    Read in a CSV file which will assign labels to structures that needs a correction in their label. 
    This can be due to wrongly annoted data from the clinic
    Inputs:
        CSV file with structure and corrected label (tab spaced)
    Returns:
        Python dictionary with stucture name and corrected label
    """
    # Create a dictionary from the correctionLabelsCSV file
    # This file contains corrected labels for all structure mask objects defined in the CSV file
    # It can be used for dictionary lookup 
    # Insert all data in the dictionary as lower case to not be dependent on such writing errors. 
    with open(correctionLabelsCSV, mode='r') as infile:
        reader = csv.reader(infile)
        correctionLabelsDictionary = {rows[0].lower():rows[1] for rows in reader}
    # Return the dictionary
    return correctionLabelsDictionary


def patientDataCopy(AllStructsArray, bodyStruct, AllStructuresOfInterest, folderOfInterest, patient, folderOutput, labelVector): 

    """
    Collect the Nifty file of interest for each patient 
    and copy it to organ folder. Also copy everything to common folder depending on 2D or 3D.
    Also create: 
    Linear combinations between all organs and the organ of interest. 
    This is done for the center of mass slice. 
    Also output a label file with label.
    Also output the volume for the 3D structure.
    Inputs:
        AllStructsArray, bodyStruct, AllStructuresOfInterest, folderOfInterest, patient, folderOutput, labelVector
    Returns:
    """
    AllDataFolder3D = 'AllData3D'
    AllDataFolder2D = 'AllData2D'

    for structNr, currStructure in enumerate(AllStructuresOfInterest):
        if not os.path.exists(folderOutput + '/' + currStructure + '_' + str(labelVector[structNr])):    
            # Create directories
            os.makedirs(folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]))
        if not os.path.exists(folderOutput + '/' + AllDataFolder2D):    
            # Create directory
            os.makedirs(folderOutput + '/' + AllDataFolder2D)
        if not os.path.exists(folderOutput + '/' + AllDataFolder3D):    
            # Create directory
            os.makedirs(folderOutput + '/' + AllDataFolder3D)

        # Get mask name and file path
        maskFileName, filenamePath = getFileNamePathForStructure(currStructure, specialStructs, folderOfInterest, patient)
        # Temp debug
        # continue
        # Define output path 
        filenamePathOut = folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_3D' + '.nii.gz'
        filenamePathOutAllDataFolder3D = folderOutput + '/' + AllDataFolder3D + '/' + patient + '_' + maskFileName + '_3D' + '.nii.gz'

        # Check if the filename exists for the specific structure
        if os.path.isfile(filenamePath) == 1: 
            # Copy the file to the correct organ folder
            shutil.copyfile(filenamePath, filenamePathOut) 
            # Copy the file to the all data 3D folder
            shutil.copyfile(filenamePath, filenamePathOutAllDataFolder3D) 
            # Get nii and numpy data from nii file
            currentNiiImage,currentStructData = PreProcess.nii2nparray(filenamePath)
            # Get projected binary 2D transversal slice from 3D volume
            tra2DProj= getBinaryArray2DProjection(currentStructData,'tra','inDataRotatedTrue')
            # Get projected binary 2D coronal slice from 3D volume
            cor2DProj= getBinaryArray2DProjection(currentStructData,'cor','inDataRotatedTrue')
            # Get projected binary 2D sagital slice from 3D volume
            sag2DProj= getBinaryArray2DProjection(currentStructData,'sag','inDataRotatedTrue')

            # Get voxel size of the current image as 3 dim vector and QA it
            voxelSize = currentNiiImage.header.get_zooms()
            assert len(voxelSize) == 3
            # Get the volume of the current 3D structure
            # As we are dealing with binary mask images, we can sum the signal of 1 valued voxels
            # and then multiply with voxel size, defined in mm 
            currentStructVolume = round(np.sum(currentStructData[:]) * voxelSize[0] * voxelSize[1] * voxelSize[2])
            # Write volume to CSV file with Pandas dataframe 
            dfVolume = pd.DataFrame([int(currentStructVolume)])
            # Get the center slice positions for the currentStructData
            # These are calculated as the array is inputed. Remember that data is flipped 90 degrees around the slice (inferior-superior) axis. 
            # Transversal slices are however not affected. 
            centerRowSlice,centerColSlice,centerTraSlice=getBinaryArrayCoM(currentStructData)
            # Calculate AddMap combinations between all structures and the current structure
            # Get signal truncation values
            signalTruncBodyAndOtherAddMap = getSignalTruncTh('BodyAndOtherAddMap')
            # Create the new array (BodyAndOtherAddMap) = allstructs - struct of interest + Body
            BodyAndOtherAddStructArray = BodyAndOtherAddMap(bodyStruct, AllStructsArray, currentStructData, signalTruncBodyAndOtherAddMap)

            # Get label data 
            currLabel = labelVector[structNr]
            # If data have been annotated incorrectly it can be corrected here by a dictionary lookup.
            # This will propagate down below for all label databases. 
            # If the entry exist in the dictionary (loaded in lowercase), assign a new corrected label, all checks performed in lower cases. 
            currStructureName = patient + '_' + maskFileName
            if currStructureName.lower() in correctionStructureLabelsDict:
                currLabel = correctionStructureLabelsDict.get(currStructureName.lower())
            # Write label to csv file with Pandas dataframe 
            df = pd.DataFrame([int(currLabel)])

            # Define shorter file path names for 2D and 3D
            filePartialNamePath2D = folderOutput + '/' + AllDataFolder2D + '/' + patient + '_' + maskFileName
            filePartialNamePath3D = folderOutput + '/' + AllDataFolder3D + '/' + patient + '_' + maskFileName

            # Write out Nifty data for the 2D tra, cor and sag and label data to the respective organ folder.
            # This include projected data. 
            # Ok to use the currentNiiImage affine for for orientation. The coronal image will be upside down in Niftynet due to
            # the fact that we do not edit the affine information. However OK when we read the data. 
            # Tra
            saveArray2nii2DTra(currentStructData, currentNiiImage, centerTraSlice, folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_2DTraCoM.nii.gz')
            saveNpSlice2nii(tra2DProj, currentNiiImage, folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_2DTraProj.nii.gz') 
            # Cor
            saveArray2nii2DCor(currentStructData, currentNiiImage, centerColSlice, folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_2DCorCoM.nii.gz')
            saveNpSlice2nii(cor2DProj, currentNiiImage, folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_2DCorProj.nii.gz') 
            # Sag
            saveArray2nii2DSag(currentStructData, currentNiiImage, centerRowSlice, folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_2DSagCoM.nii.gz')
            saveNpSlice2nii(sag2DProj, currentNiiImage, folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_2DSagProj.nii.gz') 
            # Label           
            df.to_csv(folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_Label.txt', header=None, index=False)
            
            # Write data and label and combination map to 2D AllDataFolder
            # Tra
            saveArray2nii2DTra(currentStructData, currentNiiImage, centerTraSlice, filePartialNamePath2D + '_2DTraCoM.nii.gz')
            saveNpSlice2nii(tra2DProj, currentNiiImage, filePartialNamePath2D + '_2DTraProj.nii.gz')
            # Cor
            saveArray2nii2DCor(currentStructData, currentNiiImage, centerColSlice, filePartialNamePath2D + '_2DCorCoM.nii.gz')
            saveNpSlice2nii(cor2DProj, currentNiiImage, filePartialNamePath2D + '_2DCorProj.nii.gz')
            # Sag
            saveArray2nii2DSag(currentStructData, currentNiiImage, centerRowSlice, filePartialNamePath2D + '_2DSagCoM.nii.gz')
            saveNpSlice2nii(sag2DProj, currentNiiImage, filePartialNamePath2D + '_2DSagProj.nii.gz')
            # Label
            df.to_csv(filePartialNamePath2D + '_Label.txt', header=None, index=False)
            # AddMap
            saveArray2nii2DTra(BodyAndOtherAddStructArray, currentNiiImage, centerTraSlice, filePartialNamePath2D + '_BodyAndOtherAdd2D.nii.gz')  
            
            # Write out label and combination image for the 3D volumes to the 3D AllDataFolder. Rest of the 3D data is produced in the beginning. 
            df.to_csv(filePartialNamePath3D + '_Label.txt', header=None, index=False)
            saveArray2nii3D(BodyAndOtherAddStructArray, currentNiiImage, filePartialNamePath3D + '_BodyAndOtherAdd3D.nii.gz')
            # Write out the 3D structure volume to the 3D AllDataFolder and to the respective organ folder
            dfVolume.to_csv(filePartialNamePath3D + '_3DVolume.txt', header=None, index=False)
            dfVolume.to_csv(folderOutput + '/' + currStructure + '_' + str(labelVector[structNr]) + '/' + patient + '_' + maskFileName + '_3DVolume.txt', header=None, index=False)
            

# ### Data Creation Start ### 
# Define paths
ProjectPath = '/mnt/mdstore1/Christian/Projects/StructFinder'
organ = 'Prostate'
version = 43
# Load config
modelInputConfig = loadModelInputConf(organ)
# Downsample resolution
downSampleResolution = modelInputConfig.useResolution
# Folder for inut data
RawDataInput = ProjectPath + '/' + 'dataset/RAW' + '/' + 'ProstateDataBigRawSortedNoTestData_Nifti'
# RawDataInput = ProjectPath + '/' + 'dataset/RAW' + '/' + 'Debug'
datasetFolderOut = ProjectPath + '/' + 'dataset/Final' + '/' + 'StructFinderDataset' + organ + 'V' + str(version)
infoDataInput = ProjectPath + '/' + 'dataset/RAW' + '/' + 'ProstateDataBigRawSortedNoTestDataInfo'
# Define folders of interest
folderOfInterest = RawDataInput
folderOutput = datasetFolderOut

# Get all patient folders
patFolders = os.listdir(folderOfInterest)
# Create list of structures of interest to loop over
# This list is created in lower letters
AllStructuresOfInterest, structs, specialStructs, AllLabelsOfInterest =  loadStructuresOfInterest(organ,'training') 
# Create list of structures to ignore
ignoreStructures = loadIgnoreStructures(organ)
# Load corrections for structure labels as a Python dictionary
correctionStructureLabelsDict = loadCorrectionStructureLabels(infoDataInput + '/' + 'correctionLabels.csv')

def patLoop(patNr, patient): 
    """
    Copy each structure Nifti file to each organ folder for each patient
    Also create organ and AddAllMap images 
    """
    # Collect all the patient structure available 
    # on top of each other (except Body). This does not consider any inclusion list.
    # But make sure to exclude certain structures (such as help structures, see ignore list)
    AllStructsArray, bodyStruct = collectAllStructsAvailable(ignoreStructures, folderOfInterest, patient)
    # Copy patient data and calculate AddAllMap images
    patientDataCopy(AllStructsArray, bodyStruct, AllStructuresOfInterest, folderOfInterest, patient, folderOutput, AllLabelsOfInterest)

# Count number of CPUs
nrCPU = multiprocessing.cpu_count()
if len(patFolders) < nrCPU: 
    nrCPU = len(patFolders)
#nrCPU = 4 # For test
# Init parallell job
Parallel(n_jobs=nrCPU, verbose=10)(delayed(patLoop)(patNr, patient) for patNr, patient in enumerate(patFolders))
### Data Creation End ### 


### Downsample and Create Label Database File ###
# Create downsampled version of the dataset files
# This avoids downsampling in the data loader for training.
# Also create a database CSV file of the objects and corresponding label
# Define 2D data folder name
AllDataFolder2D = 'AllData2D'
# Define input folder
originalFileFolderPath = folderOutput + '/' + AllDataFolder2D
# Define output folder
outFolderDownSampledPath = folderOutput + '/' + AllDataFolder2D + '_' + str(downSampleResolution[0]) + '_' + str(downSampleResolution[1])
# Create folder for outFolderDownSampled
if not os.path.exists(outFolderDownSampledPath):
    os.mkdir(outFolderDownSampledPath)
# Get all datafiles listed in the dataset
originalDatasetFiles = os.listdir(originalFileFolderPath)
# Get label files only
datasetFilesLabels = [file for file in originalDatasetFiles if 'Label' in file]
# Get data files only (other files)
datasetFiles = [file for file in originalDatasetFiles if 'Label' not in file]

# Define file path for ObjectLabelDatabase file
FileLabelDatabasePath = datasetFolderOut + '/' + 'ObjectLabelDatabase.csv'
# Remove old ObjectDatabaseFile
if os.path.exists(FileLabelDatabasePath): 
    os.remove(FileLabelDatabasePath)
    print("Removed old ObjectDatabaseFile")
# Define file object with appendable writing
outLabelDatabaseFileObject = open(FileLabelDatabasePath, "a")	

# Loop below has 2 purposes
# 1. Copy label files to downsample folder
# 2. Read the label file and get the name and label of the file
# Insert these in the ObjectLabelDatabase file which can be used to be converted to a dictionary later
# This makes CV file creation much faster than depending on multiple txt label reads
# This loop should NOT be parallellized as we are appending data to the same file over and over for each label file
# For every labelFile
for labelFile in datasetFilesLabels:
    # Get label path
    labelFilePath = originalFileFolderPath + '/' + labelFile
    # Copy the file to downsample folder 
    shutil.copy(labelFilePath, outFolderDownSampledPath)
    # Section following is for the ObjectLabelDatabase file
    # Read the label file with Pandas dataframe, using a C engine for increased speed
    df = pd.read_csv(labelFilePath, header=None, engine='c')
    # Extract the label value
    label = df[0].values[0]
    # Get the corresponding object name by removing "_Label.txt" from file name
    objectName = labelFile.replace('_Label.txt', "")
    # Write the data to ObjectLabelDatabase file with a new line 
    outLabelDatabaseFileObject.write(objectName + ',' + str(label) + '\n')
# Close file write for label database
outLabelDatabaseFileObject.close() 

# Read the ObjectLabelDatabase file and create dictionary for labels
with open(FileLabelDatabasePath, mode='r') as infile:
    reader = csv.reader(infile)
    ObjectLabelDataBaseDict = {rows[0]:rows[1] for rows in reader} # Keys:value

# Define downsample function
def downSampleFile(file): 
    # Get downsampled np data from Nifty file                       
    npDataDownSampled, niimgdata = PreProcess.loadAndResize(originalFileFolderPath + '/' + file, downSampleResolution[0], downSampleResolution[1])
    # Write new downsampled file in float 32 format
    saveNpSlice2nii(npDataDownSampled.astype('float32'), niimgdata, outFolderDownSampledPath + '/' + file)

# Init parallell job for downsampling
Parallel(n_jobs=nrCPU, verbose=10)(delayed(downSampleFile)(file) for file in datasetFiles)
### Downsample and Create Label Database File End ###


### This section creates volume CSV database file and dictionary for that ###
# Section was added in a later stage of development and therefore separated
# Define 3D data folder name
AllDataFolder3D = 'AllData3D'
# Define file path for ObjectVolumeDatabase file
FileVolumeDatabasePath = datasetFolderOut + '/' + 'ObjectVolumeDatabase.csv'
# Remove old VolumeDatabaseFile
if os.path.exists(FileVolumeDatabasePath): 
    os.remove(FileVolumeDatabasePath)
    print("Removed old VolumeDatabaseFile")
# Define file object with appendable writing
outVolumeDatabaseFileObject = open(FileVolumeDatabasePath, "a")
# Define folder to read 3D volume files from 
volumeFileFolderPath = folderOutput + '/' + AllDataFolder3D
# Get all volume datafiles listed 
All3DDataFiles = os.listdir(volumeFileFolderPath)
# Get volume files only
datasetFilesVolumes = [file for file in All3DDataFiles if '3DVolume' in file]

# For every volumeFile
for volumeFile in datasetFilesVolumes:
    # Get volume path
    volumeFilePath = volumeFileFolderPath + '/' + volumeFile
    # Section following is for the ObjectVolumeDatabase file
    # Read the volume file with Pandas dataframe, using a C engine for increased speed
    dfReadVolume = pd.read_csv(volumeFilePath, header=None, engine='c')
    # Extract the volume value
    volume = dfReadVolume[0].values[0]
    # Get the corresponding object name by removing "_Label.txt" from file name
    objectName = volumeFile.replace('_3DVolume.txt', "")
    # Get the label (observe that this is requested from the previously created label dictionary)
    # Object names are however the same
    label = ObjectLabelDataBaseDict.get(objectName)
    # Write the data to ObjectVolumeDatabase file with a new line (volume and label)
    outVolumeDatabaseFileObject.write(objectName + ',' + str(volume) + ',' + str(label) + '\n')
# Close file write for label database
outVolumeDatabaseFileObject.close() 

# Read the ObjectVolumeDatabase file and create dictionary 
with open(FileVolumeDatabasePath, mode='r') as infileVolume:
    readerVolume = csv.reader(infileVolume)
    ObjectVolumeDataBaseDict = {rows[0]: [rows[1],rows[2]] for rows in readerVolume} #key:values


### HDF5 database creation for image and label data ###
# Get the object names in the list
# This will be our reference to our file ID
objectList = ObjectLabelDataBaseDict.keys()
# Greate the HDF5 database file
# Every object is defined as a separate group
# Under each group there are multiple datasets created
# Best to avoid parallellization here I think
with h5py.File(datasetFolderOut + '/' + 'ObjectImageLabelDatabase.hdf5', 'w') as f:
    for index, obj in enumerate(objectList): 
        # Create a group for each object
        currPat = f.create_group(obj)
        # Create datasets for each image and label
        # Store transversal structure CoM data (0 or 1)
        traStructureCoMData = nib.load(outFolderDownSampledPath + '/' + obj + '_2DTraCoM.nii.gz').get_fdata()
        currPat.create_dataset('TraStructureCoM', data=traStructureCoMData, dtype='int8')
        # Store transversal structure projection data (0 or 1)
        traStructureProjData = nib.load(outFolderDownSampledPath + '/' + obj + '_2DTraProj.nii.gz').get_fdata()
        currPat.create_dataset('TraStructureProj', data=traStructureProjData, dtype='int8')
        # Store AddMap data (continues values between 0 and 1)
        AddMapData = nib.load(outFolderDownSampledPath + '/' + obj + '_BodyAndOtherAdd2D.nii.gz').get_fdata()
        currPat.create_dataset('AddMap', data=AddMapData, dtype='float32')
        # Store coronal structure CoM data (0 or 1)
        corStructureCoMData = nib.load(outFolderDownSampledPath + '/' + obj + '_2DCorCoM.nii.gz').get_fdata()
        currPat.create_dataset('CorStructureCoM', data=corStructureCoMData, dtype='int8')
        # Store coronal structure projection data (0 or 1)
        corStructureProjData = nib.load(outFolderDownSampledPath + '/' + obj + '_2DCorProj.nii.gz').get_fdata()
        currPat.create_dataset('CorStructureProj', data=corStructureProjData, dtype='int8')
        # Store sagital structure CoM data (0 or 1)
        sagStructureCoMData = nib.load(outFolderDownSampledPath + '/' + obj + '_2DSagCoM.nii.gz').get_fdata()
        currPat.create_dataset('SagStructureCoM', data=sagStructureCoMData, dtype='int8')
        # Store sagital structure projection data (0 or 1)
        sagStructureProjData = nib.load(outFolderDownSampledPath + '/' + obj + '_2DSagProj.nii.gz').get_fdata()
        currPat.create_dataset('SagStructureProj', data=sagStructureProjData, dtype='int8')
        # Store the label (integer depending on class)
        label = ObjectLabelDataBaseDict.get(obj)
        currPat.create_dataset('Label', data=int(label), dtype='int8')
### HDF5 database creation End ###


### HDF5 database (separate) creation for Volume and Label data ###
# A separate database was created due to the fact that we do not define 3D volume
# for the pm augmentated slices, only 3D volumes. 
# Read the ObjectVolumeDatabase file
# Get only the object names in the list
# This will be our reference to our file ID
objectListVolume = ObjectVolumeDataBaseDict.keys()
# Greate the HDF5 database file
# Every object is defined as a separate group
# Under each group there are two datasets (Volume and Label)
# Best to avoid parallellization here I think
with h5py.File(datasetFolderOut + '/' + 'ObjectVolumeLabelDatabase.hdf5', 'w') as f_volume:
    for index, objVolume in enumerate(objectListVolume): 
        # Create a group for each object
        currPat = f_volume.create_group(objVolume)
        # Create datasets for volume and label
        # Store volume data
        volume = ObjectVolumeDataBaseDict.get(objVolume)[0]
        currPat.create_dataset('Volume', data=volume)
        # Store the label data 
        label = ObjectVolumeDataBaseDict.get(objVolume)[1]
        currPat.create_dataset('Label', data=int(label), dtype='int8')
### HDF5 database creation End ###


print('Program has finished!')
