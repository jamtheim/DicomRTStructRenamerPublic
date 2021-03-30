# Author: Christian Jamtheim Gustafsson, PhD

import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import shutil
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from natsort import natsorted
import sys
import nibabel as nib
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import csv


def writeDataSplitFile(writeOutNames, dataSetType, outFileName, ObjectLabelDatabaseDict): 

    """ 
    This function writes out data line by line to a text file
    Name of subject, training or validation and label
    Uses object label lookup through dictionary (very fast)
    """
	# Display message of file written if not existing
    if os.path.exists(outFileName) == 0:
	    print("Creating new file " + outFileName)
    
    # Define file object (appendable writing)
    outFileObject = open(outFileName, "a")	
	# For every name in the input list, write out data on a new line in the file
    for currName in writeOutNames: 
        # Get label from ObjectLabelDatabse dictionary
        label = ObjectLabelDatabaseDict.get(currName)
        # Fix (if label =! None) for adapting to situations when you want to train with < 100% of the data
        # Label is set to None when currName is not found in the dictionary. 
        # The dictionary is based on the label CSV file, which has been modified by sorting per patient and data has been then been eliminated. 
        # currName on the other name is based on os.listdir(dataDir). 
        if label != None: 
            # Write data
            outFileObject.write(currName + ',' + dataSetType + ',' + str(label))
            # Insert new line
            outFileObject.write("\n")
  
	# Close file write
    outFileObject.close()

    
def generateCSVFiles(dataDir, ObjectLabelDatabaseFilePath, outputDir, nbr_cross_vals, useAugData, valSingleSplit, repeatRoundsTraining): 

    """ 
    Create CSV files for cross validation from the contents of the dataDir specified.
    Cross validation separation is based on patients. 
    This makes sure no data leakage occur for the same patient, as different structures occurs through AddMap. 
    This could be a problem if organ 1 on patient 1 is in training set and organ 2 on patient 1 is in validation set (will not happen now).  
    A dictionary of object and corresponding label is also inputed. 
    """
    # Label suffix
    labelSuffix = '_Label.txt'
    # Output file name prefix and suffix
    outFileNameBase = "dataset_split_cross_"
    outFileNameType = ".csv"
    # Check that output directory truly exist
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    # Remove old files first, avoid append data to old files
    # Checks for 100 times as many files than are created from the current maximum fold (CV) number
    # Range ends in value-1, therefore add 1
    for crossVal in range(1,(nbr_cross_vals+1)*100): 
        # Define name of output file
        outFileName = outputDir + '/' + outFileNameBase + str(crossVal) + outFileNameType
        # Check if it exists and remove it
        if os.path.exists(outFileName): 
            os.remove(outFileName)
            print("Removed old file " + outFileName)

    # List all available directory files
    dataFilesRaw = os.listdir(dataDir)
    # Data dir contains many difference files for each subject (multiple data files for each patient and one label for each data file).
    # Select only files with the _Label suffix, these are unique. 
    dataFilesAll = [subject for subject in dataFilesRaw if labelSuffix in subject]
    # All items in the list should now be unique, check this
    flag = len(set(dataFilesAll)) == len(dataFilesAll) 
    if flag == False:
        sys.exit("Not unique label files detected in cross validation creation process")
    # Then remove the file suffixes
    dataFilesAll = [subject.replace(labelSuffix, '') for subject in dataFilesAll]

    # If data augmentation has been used slices above and below the center slice has been created. 
    # Remove these from this list and add to its own list. 
    # Get list of plus and minus augmented subjects
    plusList =  [subject for subject in dataFilesAll if '_plus' in subject]
    minusList = [subject for subject in dataFilesAll if '_minus' in subject]
    # Fuse the lists
    DataListAug = plusList + minusList
    # Remove plus and minus subjects and create new subject list without them 
    DataListNoAug = [subject for subject in dataFilesAll if subject not in DataListAug]

    # As we want to create the cross validations split on patient basis get the patient name prefix
    patients = [subject.split('_mask')[0] for subject in DataListNoAug]
    # Get only unique patients 
    patients = list(dict.fromkeys(patients))

    # Create a dictionary from the ObjectLabelDatabase file
    # This file contains labels for all objects generated in the dataset
    # It can therefore always be used for lookup, independent of the CV. 
    with open(ObjectLabelDatabaseFilePath, mode='r') as infile:
        reader = csv.reader(infile)
        ObjectLabelDatabaseDict = {rows[0]:rows[1] for rows in reader}
       

    def addPatientStructureData(trainPatients,validationPatients,DataListNoAug,DataListAug,useAugData):
        """
        From the patient names add all available structure data.
        If augmentation is enabled also add that data to the 
        training set. 
        """
        # Make sure there are no overlap between train and validation patients 
        overlap = [subject for subject in trainPatients if subject in validationPatients]
        # Check if overlap is empty, it should be
        if len(overlap) != 0: 
            raise ValueError('There is overlap between training and validation patients!')
        
        # Add structures for training data, match on patient name
        trainStructures = [structure for structure in DataListNoAug if structure.split('_mask')[0] in trainPatients]
        # Add structures for validation data, match on patient name
        validationStructures = [structure for structure in DataListNoAug if structure.split('_mask')[0] in validationPatients]
        # If data augmentation is enabled
        if useAugData:
            trainStructuresAug = [structure for structure in DataListAug if structure.split('_mask')[0] in trainPatients]
            trainStructures = trainStructures + trainStructuresAug

        # Make sure there are no overlap between train and validation structures
        overlap = [structure for structure in trainStructures if structure in validationStructures]
        # Check if overlap is empty, it should be
        if len(overlap) != 0: 
            raise ValueError('There is overlap between training and validation data!')

        # Shuffle the order of data
        random.shuffle(trainStructures)
        random.shuffle(validationStructures)
        return trainStructures, validationStructures 


    if nbr_cross_vals > 1: 
        # Define folds
        kf = KFold(n_splits=nbr_cross_vals, random_state=None, shuffle=True)
        # Init crossValCounter
        crossValCounter = 0
        # This could be parallellized with enumerate and then use index as crossValCounter
        for train_index, val_index in kf.split(patients):
            # Assign training and validation patients
            trainPatients= [patients[i] for i in train_index]
            validationPatients = [patients[i] for i in val_index]
            # Add the structure files/names to the data and augmented versions if enabled
            # Lists are shuffled here
            trainStructures, validationStructures  = addPatientStructureData(trainPatients,validationPatients,DataListNoAug,DataListAug,useAugData)
            # Define output file
            outFileName = outputDir + '/' + outFileNameBase + str(crossValCounter+1) + outFileNameType
            # Write the data to file
            writeDataSplitFile(trainStructures, 'training', outFileName, ObjectLabelDatabaseDict)
            writeDataSplitFile(validationStructures, 'validation', outFileName, ObjectLabelDatabaseDict)
            # Step crossValCounter            
            crossValCounter += 1

    # If no cross validation is used, produce one split file. 
    elif nbr_cross_vals==1: 
        # Check split size
        if valSingleSplit > 0: 
            trainPatients, validationPatients = train_test_split(patients, test_size=valSingleSplit, shuffle=True)
        elif valSingleSplit == 0: 
            trainPatients = patients.copy()
            validationPatients = []
        # Add the structure files/names to the data and augmented versions if enabled
        # Lists are shuffled here
        trainStructures, validationStructures  = addPatientStructureData(trainPatients,validationPatients,DataListNoAug,DataListAug,useAugData)
        # Take into account repeatRoundsTraining
        # Reset counter
        repeatCounter = 0
        # Loop over number of times to produce all cross validation files
        for i in range(0,repeatRoundsTraining):
            # Define output file
            outFileName = outputDir + '/' + outFileNameBase + str(repeatCounter+1) + outFileNameType
            # Write the data 
            writeDataSplitFile(trainStructures, 'training', outFileName, ObjectLabelDatabaseDict)
            writeDataSplitFile(validationStructures, 'validation', outFileName, ObjectLabelDatabaseDict)
            # Step repeatCounter            
            repeatCounter += 1