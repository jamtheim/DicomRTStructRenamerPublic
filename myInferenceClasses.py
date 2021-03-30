# Author: Christian Jamtheim Gustafsson, PhD

from myPreProcessingClass import myPreProcessing
from sharedFunctions import (saveArray2nii2DTra, saveArray2nii2DCor, 
getBinaryArrayCoM, QACheckMatrixSize, BodyAndOtherAddMap, getBinaryArray2DProjection, saveNpSlice2nii)
import os
import numpy as np
from commonConfig import loadModel, getSignalTruncTh
from collections import Counter 
import csv


class myInferencePrepare:

    def __init__ (self): 
        pass


    def majorityVoting(self, ListIn): 
        """
        Majority voting, get the most frequent value in the input list
        input: list
        output: value, frequency and models (positions) in ListIn that agree
        The positions will correspond to the models assessed
        """
        occurenceCount = Counter(ListIn) 
        mostCommon = occurenceCount.most_common(1)[0][0] 
        freq = occurenceCount.most_common(1)[0][1] 
        modelsAgreeing = [i for i,x in enumerate(ListIn) if x==mostCommon]
        # If the "most common" frequency was equal to only 1 time
        if occurenceCount.most_common(1)[0][1] == 1:
            mostCommonComment = 'Maximum occurence frequency was only 1 time for each label, none label selected for majority.'
            print(mostCommonComment)
            # Exit loop
            freq = 1
            mostCommon = float("NaN")
            modelsAgreeing = float("NaN")

        return mostCommon, freq, modelsAgreeing


    def predictAndVote(self, X, AllModels):  
        """
        Predict label for input X using all models and perform majority voting. 
        Outputs most common label, frequency and probability from each model that agreed
        """   
        # Init variables
        maxProbAll = []
        maxIndexAll = []
        predProbAll = []
        # Get prediction from input using all models 
        for currCV in range(len(AllModels)): 
            # predProb contains the probability for all classes
            predProb = AllModels[currCV].predict(X)   
            # Store in array
            predProbAll.append(predProb) 
            # Get maximum probability value for current CV
            maxProb = np.amax(predProb)
            # Store in array
            maxProbAll.append(maxProb)
            # Get index for maximum probability value for current CV
            maxIndex= np.argmax(predProb)
            # Store in array
            # This will contain the label codes from all models when all iterations is finished
            maxIndexAll.append(maxIndex)

        # Majority voting if multiple models have been used
        if len(AllModels) > 1: 
            # Get most common index and frequency
            mostCommonIndex, freq, modelsAgreeing = self.majorityVoting(maxIndexAll)
            # Collect the probabilities where mostCommonIndex has highest probability for the agreeing models 
            # Init variable
            collProbMajorityAll = []
            # Convert to numpy
            predProbAll = np.array(predProbAll)
            # Loop over agreeing models and get value
            for currModel in modelsAgreeing: 
                collProbMajority = predProbAll[currModel][0][mostCommonIndex]
                # Store in array
                collProbMajorityAll.append(collProbMajority)

        elif len(AllModels) == 1: 
            mostCommonIndex = maxIndexAll[0]
            freq = 1
            modelsAgreeing = 0
            # Convert to numpy
            predProb = np.array(predProb)
            collProbMajorityAll = predProb[0][mostCommonIndex]

        return mostCommonIndex, freq, collProbMajorityAll, modelsAgreeing


    def getLargestFile(self, folderOfInterest): 
        """
        Input: Directory
        Get the largest mask file in a directory 
        Output: The file name
        """
        # Get files
        folderFiles = os.listdir(folderOfInterest)
        # Selct only files that has prefix 'mask_'
        folderFiles = [file for file in folderFiles if file.startswith('mask_')]
        # Get file sizes
        fileSizes = [os.path.getsize(folderOfInterest + '/' + file) for file in folderFiles]
        # Get maximum file size
        maxFileSize = np.amax(fileSizes)
        # Get index of that entry in the list
        maxFileSizeIndex = np.argmax(fileSizes)
        # Get the file from original list
        largestFile = folderFiles[maxFileSizeIndex]
        # Return file
        return largestFile, maxFileSize


    def CreateStructureFusion(self, filesOfInterest, folderOfInterest, patient):
        """
        Collect all structures of the patient and sum them up in one array.
        Body is excluded and outputed in a separate array. 

        Inputs:

        Returns:
        AllStructsArray, dataBody
        """
        # Import librarys for parallell processing
        from joblib import Parallel, delayed
        import multiprocessing
        # Init PreProcess class instance
        PreProcess = myPreProcessing()
        print('Creating AllStructsArray array using...')
        # Get the largest mask file. This should correspond to BODY mask
        largestFile, largestFileSize = self.getLargestFile(folderOfInterest)
        maskFileNameBody = largestFile
        # Remove file extention for compability 
        maskFileNameBody = maskFileNameBody.replace('.nii.gz','')
        # Print message
        print('BODY file was assumed to be ' + str(maskFileNameBody + '.nii.gz'))    
        # File should be mask_BODY.nii.gz. Check that this is the case  
        if maskFileNameBody != 'mask_BODY':
            print('Selected BODY file ' + str(maskFileNameBody) + ' might not be correct')
        # Get the full path of the assumed BODY file
        filenamePathBody = folderOfInterest + '/' + maskFileNameBody + '.nii.gz'
        # Get nii and body array data from nii file
        niiBody, dataBody = PreProcess.nii2nparray(filenamePathBody)
        # QA Check for matrix size
        QACheckMatrixSize(dataBody)
        
        # Define function for reading the data file
        def getData(file, folderOfInterest): 
            # If not BODY structure read the structure
            if file != maskFileNameBody + '.nii.gz':
                # Get nii and body array data from nii file
                niiData, data = PreProcess.nii2nparray(folderOfInterest + '/' + file)
            else: # If it is body, assign just zeroes as body should not give signal to this sum array
                data = np.zeros(dataBody.shape)
            return data

        # Count number of CPUs and assign whats needed
        nrCPU = multiprocessing.cpu_count()
        if len(filesOfInterest) < nrCPU: 
            nrCPU = len(filesOfInterest)
        # Init parallell job. This is equal and faster compared to old method
        data = Parallel(n_jobs=nrCPU, verbose=10)(delayed(getData)(file, folderOfInterest) for index, file in enumerate(filesOfInterest))                  
        # First dimension contain all the structures from the parallell run, sum over it
        AllStructsArray = np.sum(data,axis=0)  

        return AllStructsArray, dataBody
   

    def CreateInferenceData(self, AllStructsArray, bodyStructArray, fileOfInterest, folderRead, folderWrite):
        """
        Create a nii.gz file with all structures but the current (including body) for the center of mass slice
        Will also write the organ files for tra, cor and sag projections. 
        This must depend on how training data is created
        This must be in accordance with the function patientDataCopy
        used for creating training data.
        The details need to be the same for these functions. 

        Inputs:
        AllStructsArray, bodyStruct, fileName, folderRead, folderWrite    
        Returns:
        Writes the current structure tra, cor and sag projection data and BodyAndOtherAddMap 
        for the center of mass slice slice
        Also returns code for success or not and the file paths written. 
        """
        # Init PreProcess class instance
        PreProcess = myPreProcessing()

        # Load data for current file
        try:
            # Try loading data
            currentNiiImage, currentStructNpData = PreProcess.nii2nparray(folderRead + '/' + fileOfInterest)
            # If file is not existing or not loading correctly set data volyme to empty, see later condition below.     
        except:
            currentStructNpData = []        

        # Continue only if currentStructNpData contains at least one voxel with value 1
        # or else impossible to center of mass slice of it. Also, we can not input empty data to the network.
        if np.sum(currentStructNpData[:]) >= 1:
            # Get threshold value for signal
            signalTruncAddBodyAndOtherAllMap = getSignalTruncTh('BodyAndOtherAddMap')
            # Create the AddMap from shared function
            BodyAndOtherAddStructArray = BodyAndOtherAddMap(
                bodyStructArray, AllStructsArray, currentStructNpData, signalTruncAddBodyAndOtherAllMap)
  
            # Redefine file name, remove file ending
            fileOut = fileOfInterest.replace('.nii.gz', '')
            # Get the center of mass slice for the currentStructNpData
            centerRowSlice,centerColSlice,centerTraSlice=getBinaryArrayCoM(currentStructNpData)
            # Get projected binary 2D transversal, coronal and sagital slice from 3D volume
            tra2DProj= getBinaryArray2DProjection(currentStructNpData,'tra','inDataRotatedTrue')
            cor2DProj= getBinaryArray2DProjection(currentStructNpData,'cor','inDataRotatedTrue')
            sag2DProj= getBinaryArray2DProjection(currentStructNpData,'sag','inDataRotatedTrue')
            # Write out Nifty data for the 2D tra, cor and sag projection slices
            filePathOut2DTraProj = folderWrite + '/' + fileOut + '_2DTraProj.nii.gz'
            filePathOut2DCorProj = folderWrite + '/' + fileOut + '_2DCorProj.nii.gz'
            filePathOut2DSagProj = folderWrite + '/' + fileOut + '_2DSagProj.nii.gz'
            saveNpSlice2nii(tra2DProj, currentNiiImage, filePathOut2DTraProj) 
            saveNpSlice2nii(cor2DProj, currentNiiImage, filePathOut2DCorProj)     
            saveNpSlice2nii(sag2DProj, currentNiiImage, filePathOut2DSagProj)

            # BodyAndOtherAddMap path
            filePathOutBodyAndOtherAdd2D = folderWrite + '/' + fileOut + '_BodyAndOtherAdd2D.nii.gz'
            saveArray2nii2DTra(BodyAndOtherAddStructArray.astype('float32'), currentNiiImage, centerTraSlice, filePathOutBodyAndOtherAdd2D)
            # Set exist flag
            imgDataExist = 1
            return filePathOut2DTraProj, filePathOut2DCorProj, filePathOut2DSagProj, filePathOutBodyAndOtherAdd2D, imgDataExist

        else:
            imgDataExist = 0
            # Set empty file path strings
            filePathOut2DTraProj = []
            filePathOut2DCorProj = []
            filePathOut2DSagProj = []
            filePathOutBodyAndOtherAdd2D = []
            return filePathOut2DTraProj, filePathOut2DCorProj, filePathOut2DSagProj, filePathOutBodyAndOtherAdd2D, imgDataExist


    def loadAllCVModels(self, modelExpFolder, nrCrossVal, modelEpochIter, organ, modelType): 
        """
        Load all models and save them in one variable
        """
        # Init AllModel as empty structure
        AllModels = []
        # Load each model from each cross validation into a separate object 
        for currCV in range(1, nrCrossVal+1): 
            # Define model folder
            modelExpCrossValFolder = modelExpFolder + '/' + 'cv' + str(currCV)
            # List directory for the currCV
            dirCurrCV = os.listdir(modelExpCrossValFolder)
            # Get the file for the specific iteration
            if modelEpochIter < 10:
                closestWeightFile = [i for i in dirCurrCV if i.lower().startswith('weights.0' + str(modelEpochIter))] 
            else: 
                 closestWeightFile = [i for i in dirCurrCV if i.lower().startswith('weights.' + str(modelEpochIter))]    
            # Quality check
            if closestWeightFile != []:  
                # Check size of it, must be maximum 1
                if len(closestWeightFile) != 1:
                    raise ValueError('More than one weight file was found')
            else: 
                # If empty 
                raise ValueError('No weight file was found')
            # Load weights from path
            weightPath = modelExpCrossValFolder + '/' + closestWeightFile[0]
            # Load model for inference and init weights
            model = loadModel('inference', weightPath, organ, modelType)
            # Append the model to models
            AllModels.append(model)
            print(str(len(AllModels)) + ' models loaded')
            print(weightPath)

        return AllModels


    def adaptROINameForMaskFile(self, ROINameDicomCollected): 
        """
        Structure names in list_rt_structs does not correspond to outputed filenames for the nii.gz files
        as some characters are removed and some character changed to something else.
        Some characters are removed, and some is replaced with a '-' sign, rules defined below.
        """       
        # removeChar = ['[', ']', '(', ')', '+', '.']
        # Added characers, note \ (escape character)
        removeChar = ['[', ']', '(', ')', '+', '.', '?', ':', '/', '\\', ',', '¨']
        # replaceChar = ' '
        replaceChar = [' ']
        changeChar = ['å','ä','ö','Å','Ä','Ö']
        changeCharToThis = ['a','a','o','A','A','O']

        # Assign ROI name to new variable for adaption
        ROINameCollectedAdapted = ROINameDicomCollected
        # Loop through all characters for removal
        for remove in removeChar:
            # If the character exist in the name remove it
            if remove in ROINameCollectedAdapted:
                ROINameCollectedAdapted = ROINameCollectedAdapted.replace(
                    remove, '')
        # Loop throu all characters that need to be replaced
        for replace in replaceChar:
            if replace in ROINameCollectedAdapted:
                ROINameCollectedAdapted = ROINameCollectedAdapted.replace(
                    replace, '-')
        # Loop throu all characters that need to be changed
        for index, change in enumerate(changeChar):
            if change in ROINameCollectedAdapted:
                ROINameCollectedAdapted = ROINameCollectedAdapted.replace(
                    change, changeCharToThis[index])

        ROINameForMaskFile = ROINameCollectedAdapted
        return ROINameForMaskFile


    def getVolumeLabelStatistics(self, objectVolumeDatabaseFilePath, numberClasses): 
        """
        Get volume statistics for all classes in the training data volume database
        This is used to sanity check the inference results.  
        """ 
        # Read the Volume Label database and create dictionary
        with open(objectVolumeDatabaseFilePath, mode='r') as infileVolume:
            readerVolume = csv.reader(infileVolume)
            ObjectVolumeDataBaseDict = {rows[0]: [rows[1],rows[2]] for rows in readerVolume} #key:values
        # Get list of volume values
        VolumeLabelList= list(ObjectVolumeDataBaseDict.values())
        # Create a class vector
        classVector = list(range(0,numberClasses))
        # Create and empty list with containers for all classes 
        volumeDataAllClasses = [[] for i in range(numberClasses)]
        # Loop through values in the list and collect volume data for every label and put it into the new list structure 
        # For every label in the class label vector
        for classLabel in classVector:
            # For every item contained in the volume label list
            for item in VolumeLabelList:
                # If the items class label is the same as the current label in outer loop
                if item[1] == str(classLabel): 
                    # Append the volume data to the list structure
                    # Its classLabel index corresponds to the label 
                    volumeDataAllClasses[classLabel].append(int(item[0])) 

        # Create array with descriptive statistics
        # First create and empty list with containers for all classes 
        medianVolumeAllClasses = [([]) for i in range(numberClasses)]
        meanVolumeAllClasses = [([]) for i in range(numberClasses)]
        stdVolumeAllClasses = [([]) for i in range(numberClasses)]
        minVolumeAllClasses = [([]) for i in range(numberClasses)]
        maxVolumeAllClasses = [([]) for i in range(numberClasses)]
        # Loop through the classes
        for classLabel in classVector:
        # Calculate volume statistic descriptives
            medianVolumeAllClasses[classLabel] = np.median(volumeDataAllClasses[classLabel])
            meanVolumeAllClasses[classLabel] = np.mean(volumeDataAllClasses[classLabel])
            stdVolumeAllClasses[classLabel] = np.std(volumeDataAllClasses[classLabel]) # std for population
            minVolumeAllClasses[classLabel] = np.min(volumeDataAllClasses[classLabel])
            maxVolumeAllClasses[classLabel] = np.max(volumeDataAllClasses[classLabel])
        # Return data         
        return meanVolumeAllClasses, stdVolumeAllClasses, minVolumeAllClasses, maxVolumeAllClasses, volumeDataAllClasses


    def checkQCVolumeInferenceNormalDist(self, currentStructVolume, label, meanVolumeAllClasses, stdVolumeAllClasses, nbrStd): 
        """
        Check if the volume of the infered object is within a defined volume interval 
        for the training data, specific for the determined label (mostCommonIndex).   
        The interval is defined as mean plus minus nbrStd std
        """    
        # Set default QC status
        QCStatus = True
        # If volume is smaller
        if currentStructVolume < (meanVolumeAllClasses[label] - nbrStd*stdVolumeAllClasses[label]): 
            QCStatus = False
        # If volume is larger 
        if currentStructVolume > (meanVolumeAllClasses[label] + nbrStd*stdVolumeAllClasses[label]): 
           QCStatus = False
        # Return data 
        return QCStatus


    def checkQCVolumeInferencePercentile(self, currentStructVolume, label, volumeDataAllClasses, percentileLow, percentileHigh): 
        """
        Check if the volume of the infered object is within a defined volume interval 
        for the training data, specific for the determined label (mostCommonIndex).   
        The interval is defined as a percentileRange 
        """  
        # Set default QC status
        QCStatus = True
        # If volume is smaller
        if currentStructVolume < np.percentile(volumeDataAllClasses[label],percentileLow):
            QCStatus = False
        # If volume is larger 
        if currentStructVolume > np.percentile(volumeDataAllClasses[label],percentileHigh):
           QCStatus = False
        # Return data 
        return QCStatus
       

    def write2log(self, logFilePath, logMessage):
        # Write to log file, insert new line and close file
        outFileObjectErrors = open(logFilePath, "a")
        outFileObjectErrors.write(logMessage)
        outFileObjectErrors.write("\n")
        outFileObjectErrors.close()

    
    def checkIfTarget(self, stringIn):
        # Set targetExist init status
        targetExist = 0
        # Convert ROIName to lower case
        stringIn = stringIn.lower()
        # Define target names of interest (lower case)
        target = ['gtv', 'ctv', 'ptv'] # Will cover ctvt, ctvn ...
        # Look for existance of target names in the ROIName
        if any(i in stringIn for i in target):
            targetExist = 1
        # Return target exist flag    
        return targetExist


    def getTargetDose(self, ROIName):
        # Extract the target dose from the ROIName
        # Import support for using regular expressiosn
        import re
        # Set delimiter
        delim = '_'
        # As a first step, split ROIName from start (left) at delim get only last bit, this excludes CTVT1 
        # and so on which has a number in it. Required existance of delimiter in name though. 
        doseLevelPart = ROIName.split(delim,1)[-1]
        # print(doseLevelPart)
        # Use try/except statement if inference falesly giving a target name but
        # there is no dose information available efter a delim (for example bladder seen as as a target)
        try:
            # Evaluate according to 
            # https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string      
            # Extracts the floating point number with respect to . and ,
            # doseLevelPartCleaned = re.findall(r"[-+]?\d*\.\d+|\d+", doseLevelPart)
            doseLevelPartCleaned = re.findall(r"[-+]?\d*[.,]\d+|\d+", doseLevelPart)
            # print(doseLevelPartCleaned)
            # If multiple values found, take the first one, is then closest to CTV_
            doseLevelPartCleaned = doseLevelPartCleaned[0]
            # print(doseLevelPartCleaned)
        except: 
            # Set string to empty if not working out
            doseLevelPartCleaned = ''

        # Return the dose level information as a string
        return doseLevelPartCleaned
