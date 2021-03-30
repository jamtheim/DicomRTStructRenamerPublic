# This script will extract/move a determined number of patients from the the raw data to a test dataset
# Author: Christian Jamtheim Gustafsson, PhD
import os
import numpy as np
import os.path
import random
import shutil


# Define what dataset to process
procDataSet = 'ProstateDataBigRawSorted'
# Patient input folder
allPatFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/' + procDataSet
# Output testDataFolder folder
testDataFolder = '/mnt/mdstore1/Christian/Projects/StructFinder/dataset/RAW/' + procDataSet + 'TestDataset'
# Define number of random patients to move to testDataFolder
testDataNrPatients = 200

# Make sure the output directory exists
if not os.path.isdir(testDataFolder):
    # Create dir
    os.mkdir(testDataFolder)

# List patients in patFolder
patFolders = os.listdir(allPatFolder)
# Shuffle the patient list (even if we use randomization later)
random.shuffle(patFolders)
# Get number of patients in the list
nrPat = len(patFolders)

# Define unique random numbers in same amount as number of requested test data patients 
# The numbers are contained within 0:nrPat
randomPatNrs = random.sample(range(0,nrPat),testDataNrPatients)
# print(randomPatNrs)
# Make sure there are equal amount of unique randomPatNrs as requested number of test set patients
assert testDataNrPatients == len(randomPatNrs)

# Loop through the patients and move/copy folders to the test data folder
for patNr, patient in enumerate(patFolders): 
    # If current patient index number exists in the randomly generated numbers, do action
    if patNr in randomPatNrs:
       destination = shutil.move(allPatFolder + '/' + patient, testDataFolder + '/' + patient) 

print("Program has completed!")