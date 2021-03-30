# Author: Christian Jamtheim Gustafsson, PhD

import os
import random
import numpy as np
import nibabel as nib
from random import seed
from random import randint
import matplotlib.pyplot as plt
from myPreProcessingClass import myPreProcessing
import psutil
import shutil
import scipy.ndimage as ndi 
import pydicom

# Init PreProcess class instance
PreProcess = myPreProcessing()


def getStructureFile(path):
    """
    Search a given path for a RT structure DICOM file
    Inputs:
        path (str): Path to the DICOM file
    Returns:
        The RT file name
    """
    files = os.listdir(path)
    structFile = [f for f in files if ".dcm" in f]

    if len(structFile) == 0:
        print('No RT structure file could be located. Make sure the file is located in the specified folder...')
        exit
        pass
    return structFile[0]


def getRTStructFile(path):
    """
    Search a given path for a RT structure DICOM file
    Inputs:
        path (str): Path to the DICOM file
    Returns:
        The RT file name
    """
    files = os.listdir(path)
    structFile = [f for f in files if ".dcm" in f]
    structFile = [f for f in files if "RS." in f]
    
    if len(structFile) == 0:
        print('No RT structure file could be located. Make sure the file is located in the specified folder...')
        exit
        pass
    return structFile[0]


def moveRTStructFile(patientPath):
    """
    Move RS Dicom file that is not a CT Dicom file to RTStruct sub folder
    """
    # Define name of this folder
    RTStructFolderName = 'RTStruct'
    # Get path of the folder
    RTStructPath = patientPath + '/' + RTStructFolderName
    # If folder does not exist
    if not os.path.isdir(RTStructPath):
        # Create dir
        os.mkdir(RTStructPath)
        # Get files in patient dir
        files = os.listdir(patientPath)
        # Move files that does not belong to the CT Dicom files
        for file in files:
            if ".dcm" in file: 
                if "CT." not in file: 
                    shutil.move(patientPath + '/' + file, RTStructPath)
        
    # Return the name of the RTStruct folder
    return RTStructFolderName


def moveCTFiles(patientPath):
    """
    Move CT DICOM files to sub directory
    """
    # Define name of this folder
    CTFolderName = 'CT'
    # Get path of the folder
    CTPath = patientPath + '/' + CTFolderName
    # If folder does not exist
    if not os.path.isdir(CTPath):
        # Create dir
        os.mkdir(CTPath)
        # Get CT files in patient dir
        files = os.listdir(patientPath)
        # Move CT files 
        for file in files:
            if ".dcm" in file: 
                if "CT." in file: 
                    shutil.move(patientPath + '/' + file, CTPath)
        
    # Return the name of the CT folder
    return CTFolderName


def printScanDate(patCTFolder):
    # Get the files
    files = os.listdir(patCTFolder)
    # Make sure they are dicom
    files = [f for f in files if ".dcm" in f]
    # Make sure CT file
    files = [f for f in files if "CT" in f]
    # Read one file
    dicomData = pydicom.dcmread(patCTFolder + '/' + files[0])
    scanDate = dicomData.AcquisitionDate
    # print(scanDate)
    return scanDate


def getFileNamePathForStructure(structureName, specialStructs, folderOfInterest, patient): 
    """
    Get the correct file name for the structure of interest
    Also adapt to different cases where specialStructrs has specific naming and a more fuzzy description can be used
    
    Inputs:
        structureName, folderOfInterest, patient
    Returns:
    maskFileName, path
    """
    # Get directory listing of current patient 
    dirCurrPatient = os.listdir(folderOfInterest + '/' + patient)
    # Check what to do 
    if structureName in specialStructs: 
        # Get an exact file name back for the special structs inputed more with a fuzzy decription.
        # Find list element that starts or ends with the structureName (always parsed in lower letters)
        closestFile = [i for i in dirCurrPatient if i.lower().startswith('mask_' + structureName) or i.lower().endswith(structureName + '.nii.gz')]    
    else: 
        # If not special structure
        # Get the file from the list where the lower letter version 
        # of the file name corresponds to the structureName exactly.
        # StructureName is parsed in with lower letter. 
        closestFile = [i for i in dirCurrPatient if i.lower() == 'mask_' + structureName + '.nii.gz']

    # If not empty
    if closestFile != []:  
        # Check size of it, must be maximum 1
        if len(closestFile) != 1:
            # Temp debug
            raise ValueError('More than one file was found for matching structures in patient ' + patient)
            # print('More than one file was found for matching structures in patient ' + patient)
    else: 
        # If empty assign dummy name 
        # Existance for this is checked later on in the code 
        closestFile = 'mask_' + 'ThisStructDoesNotExist_' + str(random.randrange(1000000000)) + '.nii.gz'
        # FileNamePathForStructure = folderOfInterest + '/' + patient + '/' + maskFileName + '.nii.gz'

    # Define maskFileName
    maskFileName = closestFile
    # Remove '.nii.gz' from the name (for the output variable)
    maskFileName = maskFileName[0].replace('.nii.gz','') 
    # Define the path
    FileNamePathForStructure = folderOfInterest + '/' + patient + '/' + maskFileName + '.nii.gz'

    return maskFileName, FileNamePathForStructure
    

def QACheckMatrixSize(dataBody): 
    """
    Check size of matrix
    """
    if dataBody.shape[0] != 512:
        # raise ValueError('Matrix has wrong size, it is not 512')
        print('Matrix has wrong size, it is not 512')
    if dataBody.shape[1] != 512:
        # raise ValueError('Matrix has wrong size, it is not 512')
        print('Matrix has wrong size, it is not 512')


def fuseStructures(existingFusion, newDataFile): 
    """
    Fuse structure data into one array.
    """
    # Get nii and numpy data from nii file
    niimgdata,npdata = PreProcess.nii2nparray(newDataFile)
    # Store data on top of each other in the same array
    # Good to know that this operation consumes most of the computing time
    existingFusion = existingFusion + npdata 
    return existingFusion


def BodyAndOtherAddMap(bodyArray, AllStructArray, currentArray, signalTruncTh): 
    """
    Fuse other organs and body into one map
    Put weights on the different parts 
    Limit signal to signalTruncTh
    """  
    # Add them with weights
    # Having 0.2 for others allows signal pileup 4 times (0.8) + body (0.1) = 0.9
    BodyAndOther = 2/10*(AllStructArray - currentArray) + 1/10*bodyArray
    # Make sure they are positive
    BodyAndOther = np.absolute(BodyAndOther)
    # Signal capitation. This will make sure the signal values does not
    # increase above a certain threshold value
    # Get index of values above threshold
    thresholdIndexes = BodyAndOther > signalTruncTh
    # Set those values equal to the threshold
    BodyAndOther[thresholdIndexes] = signalTruncTh
    return BodyAndOther


def saveArray2nii2DTra(npStructData, niiStruct, sliceOfInterest, filePath): 
    """
    Save array as nii.gz file. Only a single transversal slice is selected from the array. 
    Preserve affine information but not header as data type can be changed. 
    """  
    # Old command. This forces 8 bit format to be written even if data is floats. 
    # This is because the original header says 8 bit. Value 0.10 will be represented as 0.1012 and 0.3 as 0.3012.
    # Not sure how reproducible this is or why it is so. 
    # Should probably be avoided even if file size and loading is quicker for training. 
    # imgData = nib.Nifti1Image(npStructData[:,:,sliceOfInterest], niiStruct.affine, niiStruct.header)
    imgData = nib.Nifti1Image(npStructData[:,:,sliceOfInterest], niiStruct.affine)
    #print(npStructData.dtype)
    #plt.imshow(npStructData[:,:,sliceOfInterest]); plt.show(block=False); plt.pause(10); plt.close()
    nib.save(imgData, filePath)


def saveArray2nii2DCor(npStructData, niiStruct, sliceOfInterest, filePath): 
    """
    Save array as nii.gz file. Only a single coronal slice is selected from the array. 
    Preserve affine information but not header as data type can be changed. 
    Be aware that arrays inputed are rotated around the slice axis. This is due to the way Nifti reader works. 
    That means that the "column" index must be populated to produce a Coronal slice according
    to standard patient orientation. 
    """  
    imgData = nib.Nifti1Image(npStructData[:,sliceOfInterest,:], niiStruct.affine)
    nib.save(imgData, filePath)


def saveArray2nii2DSag(npStructData, niiStruct, sliceOfInterest, filePath): 
    """
    Save array as nii.gz file. Only a single sagital slice is selected from the array. 
    Preserve affine information but not header as data type can be changed. 
    Be aware that arrays inputed are rotated around the slice axis. This is due to the way Nifti reader works. 
    That means that the "row" index must be populated to produce a sagital slice according
    to standard patient orientation. 
    """  
    imgData = nib.Nifti1Image(npStructData[sliceOfInterest,:,:], niiStruct.affine)
    nib.save(imgData, filePath)


def saveNpSlice2nii(npData, niiStruct, filePath): 
    """
    Save np slice to nii file.
    Preserve affine information.
    """  
    imgData = nib.Nifti1Image(npData, niiStruct.affine)
    nib.save(imgData, filePath)


def saveArray2nii3D(npStructData, niiStruct, filePath): 
    """
    Save array as nii.gz file. Whole 3D volume is selected, slice of interest not needed. 
    Preserve affine information but not header as data type is changed. 
    """  
    # imgData = nib.Nifti1Image(npStructData, niiStruct.affine, niiStruct.header)
    imgData = nib.Nifti1Image(npStructData, niiStruct.affine)
    nib.save(imgData, filePath)

    
def RemoveItemsFromList(originalList, itemsToRemove): 
    """
    Remove items from input list which are defined in itemsToRemove
    Input: Original list, list items to remove
    Output: Cleaned list
    """
    # Init exist flag
    existFlag = 0

    # First make sure all is lower case for itemsToRemove as this is standard
    itemsToRemove = [each_string.lower() for each_string in itemsToRemove]
    # Copy the original list to a new variable
    # Do not use X = Y, this creates only a reference for lists.
    editedList = originalList.copy()
    # Loop through all items in input list to see if they starts with any of the objects defined in the itemsToRemove. If so, remove it from new list. 
    for i, item in enumerate(originalList): 
        if item.lower().startswith(tuple(itemsToRemove)): 
            editedList.remove(item)
            # Set exist flag
            existFlag = 1
        
    return editedList, existFlag


def checkIfProcessRunning(processName):
    '''
    Check if there is any running process that contains the given name processName.
    https://thispointer.com/python-check-if-a-process-is-running-by-name-and-find-its-process-id-pid/
    '''
    #Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    https://thispointer.com/python-check-if-a-process-is-running-by-name-and-find-its-process-id-pid/
    the given string processName
    '''
    listOfProcessObjects = []
    #Iterate over the all the running process
    for proc in psutil.process_iter():
       try:
           pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
           # Check if process name contains the given name string.
           if processName.lower() in pinfo['name'].lower() :
               listOfProcessObjects.append(pinfo)
       except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess) :
           pass
    return listOfProcessObjects


def getClassWeights(useClassWeights, useClasses, y_train): 
    """
    Calulates class weights in balanced mode if enabled
    Otherwise return none
    """ 
    from sklearn.utils import class_weight
    if useClassWeights: 
        # Get classes in a sorted array 
        uniqueClasses = np.arange(start=0, stop=useClasses, step=1)
        # Collect all labels as integers from training data
        usedClassesAll = [int(i) for i in y_train]
        # Define a class weights dictonary
        determinedClassWeights = dict(zip(uniqueClasses, class_weight.compute_class_weight('balanced', uniqueClasses, usedClassesAll))) 
        # QA check
        assert len(determinedClassWeights) == useClasses
    else: 
        determinedClassWeights = None        

    return determinedClassWeights 


def printClassWeights(myClassWeights, fileOut): 
    file = open(fileOut, "w")
    file.write(str(myClassWeights))
    file.close()


def getBinaryArrayCoM(arrayIn):
    """
    Get the coordinates for the center of mass of the binary segmentation stack.
    If the transversal slice is empty in this slice position repeat the 
    center of mass detection using a subset of the data.
    This subset consists of data defined from this empty slice until the end of the stack.
    One important thing to remember: Data in the binary stack is rotated 90 degrees, this happends when reading the Nifti file. 
    Output is given as row, column, slice however array is inputed. 
    """
    # QA - make sure incoming array is containing only binary data
    assert arrayIn.size == np.count_nonzero((arrayIn==0) | (arrayIn==1))
    # Detect center of mass of the 3D array 
    # Output is row, column, slice with respect to input (not real world orientation as array can be flipped)
    rowCoM,columnCoM,sliceCoM = ndi.center_of_mass(arrayIn) 
    # Get rounded integers as numbers out
    rowCoM = int(round(rowCoM))
    columnCoM = int(round(columnCoM))
    sliceCoM  = int(round(sliceCoM))

    # But if two signal object have had the same size, for example two nodes, the center of mass transversal slice might still be empty. 
    # See patient 184185622558 for example
    # In that case exclude the data before this position in the image data in the array 
    # and do a new center of mass detection for the rest of the data from this slice.
    # Add old reference of center of mass slice in the final calculation to get back to original coordinate system.  
    if np.sum(arrayIn[:, :, sliceCoM]) == 0: # If transversal slice does not contain any signal
        rowCoM2,columnCoM2,sliceCoM2 = ndi.center_of_mass(arrayIn[:, :, sliceCoM:-1]) 
        # Take into consideration new new coorodinate system as new center of mass is only calculated on a subset in slice direction
        # The coordinate system in other axes is unchanged. 
        rowCoM = int(round(rowCoM2))
        columnCoM = int(round(columnCoM2))
        sliceCoM = int(round(sliceCoM + sliceCoM2))
        # Check that the transversal image slice is not empty as we do not want to produce empty data
        # If empty data is detected, halt the program
        if np.sum(arrayIn[:, :, sliceCoM]) == 0:
            raise Exception('Zero-valued 2D data was produced and program was halted')

    # Convert to integer value
    rowCoM = int(rowCoM)
    columnCoM = int(columnCoM)
    sliceCoM = int(sliceCoM)

    # Return data 
    return rowCoM,columnCoM,sliceCoM


def getBinaryArray2DProjection(arrayIn,reqAnatOrient,rotatedData): 
	"""
	Get the binary projected data from a 3D input array. Projection is outputed 
	as a requested anatomical orientation, chosen as the second input, further defined along a chosen axis. 
	A third input defines if input data have been rotated or not.
	Axis are numbered 0, 1, 2 and corresonds to views along row, column and slice. 
	This is with regards to incoming array and not anatomical orientation of the data.
	With our reading of Nifti files, the data is rotated around the slice axis (inferior-superior).  
	Implementation and input in this function is therefore based on our case data. 
	"""
	# QA - make sure incoming array is containing only binary data
	assert arrayIn.size == np.count_nonzero((arrayIn==0) | (arrayIn==1))
	# QA - make sure incoming array has 3 dimensions exactly
	assert arrayIn.ndim == 3
	# Get 2D summation data depending on requested anatomical orientation and orientation of input array
	if rotatedData == 'inDataRotatedTrue':
		if reqAnatOrient == 'cor': 
			projectedSumData = arrayIn[:, :, :].sum(axis=1) #Along column
		if reqAnatOrient == 'sag': 
			projectedSumData = arrayIn[:, :, :].sum(axis=0) #Along row
		if reqAnatOrient == 'tra':
			projectedSumData = arrayIn[:, :, :].sum(axis=2) #Along slice

	if rotatedData == 'inDataRotatedFalse':
		if reqAnatOrient == 'cor': 
			projectedSumData = arrayIn[:, :, :].sum(axis=0) #Along row 
		if reqAnatOrient == 'sag': 
			projectedSumData = arrayIn[:, :, :].sum(axis=1) #Along column
		if reqAnatOrient == 'tra':
			projectedSumData = arrayIn[:, :, :].sum(axis=2) #Along slice

	# However, we are not intrested by the sum, only the extent of pixels with values
	# Create an zero valued int8 matrix with the same size
	sliceOut = np.zeros(projectedSumData.shape, dtype=np.int8)
	# The projection should be binary in output 
	# Get matrix index of where matrix values are 1 or above 
	thresholdIndexes = projectedSumData >= 1
	# Set those values equal to 1 in the sliceOut
	sliceOut[thresholdIndexes] = 1
	# QA - make sure outgoing slice contains only binary data
	assert sliceOut.size == np.count_nonzero((sliceOut==0) | (sliceOut==1))
	# Return data 
	return sliceOut

