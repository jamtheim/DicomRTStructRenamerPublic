# Author: Christian Jamtheim Gustafsson, PhD

import os
import numpy as np


class myEvaluation: 

    def __init__ (self): 
        pass


    def writeDataFileStats(self, labelFailFreq, labelFailPercentage, labelExpFreq, labelVector, filePath): 
            """ 
            This function writes out data line by line to a text file
            Writes the name of the label, fail frequency, fail frequency in percentage and frequency total occured
            """
            # Display message of file written if not existing
            if os.path.exists(filePath) == 0:
                print("Created new file for failed object statistics")

            # Define file object (appendable writing)
            outWriteFileObject = open(filePath, "a")	
            # For every object in the list, write out data on a new line in the file
            for index, currLabel in enumerate(labelVector): 
                # Write the data    
                outWriteFileObject.write(str(currLabel) + ' \t '  + str(labelFailPercentage[index]) + '%' + ' \t ' + str(labelFailFreq[index]) + ' \t ' + str(labelExpFreq[index]))
                # Insert new line
                outWriteFileObject.write("\n")
             # Close file write
            outWriteFileObject.close()


    def writeDataFileFailed(self, failedObjects, failedObjectsPredictedAs, filePath): 
            """ 
            This function writes out data line by line to a text file
            Writes the name of the failed objects and the predicted class
            Inserts new line after each writing
            """
            # Display message of file written if not existing
            if os.path.exists(filePath) == 0:
                print("Created new file for failed objects")

            # Define file object (appendable writing)
            outWriteFileObject = open(filePath, "a")	
            # For every object in the list, write out data on a new line in the file
            for index, currObject in enumerate(failedObjects): 
                # Write the name    
                outWriteFileObject.write(currObject + ' \t ' + failedObjectsPredictedAs[index])
                # Insert new line
                outWriteFileObject.write("\n")
             # Close file write
            outWriteFileObject.close()


    def writeFailedClassifications(self, model, validation_generator_predict, saveFolder, saveFileFailed, saveFileStats, InferenceStructureNames):
        """
        Write objects to file that were not classified correctly
        Also write what they were predicted as. 
        """    
        # Get probabilities for predicted labels
        probPredict = model.predict(validation_generator_predict, verbose=1)       
        # Loop over each subject (row) and extract index for max probability value
        # Index position will correspond to label in an np array
        predLabel = [int(np.argmax(i)) for i in probPredict]
        predLabel = np.asarray(predLabel)
        # Get expected labels (ground truth)
        expLabel = validation_generator_predict.labels
        # Convert to integer np array
        expLabel = [int(i) for i in expLabel]
        expLabel = np.asarray(expLabel)
        # Get index where predLabel and expLabel differ
        # We know its one dimensional so index 0 is ok. 
        notCorrectIndex = np.where(predLabel!=expLabel)[0]
        # Collect all objects with wrong classification
        failedObjects = [validation_generator_predict.list_IDs[i] for i in notCorrectIndex]  
        # Collect their prediction
        failedObjectsPredictedAs = [InferenceStructureNames[predLabel[i]] for i in notCorrectIndex] 
        # Define output file for failed objects
        outFileNamePathFailed = saveFolder + '/' + saveFileFailed
        # Define output file for statistics
        outFileNamePathStats = saveFolder + '/' + saveFileStats
        # Write the data 
        self.writeDataFileFailed(failedObjects, failedObjectsPredictedAs, outFileNamePathFailed)
        # Calculate occurance of labels available and labels not predicted correctly 
        labelFailFreq, labelExpFreq, labelFailPercentage, labelVector = self.calcStatClassifications(expLabel, notCorrectIndex)
        # Write the frequency and percentage failing for each class together with availabe items for that class
        self.writeDataFileStats(labelFailFreq, labelFailPercentage, labelExpFreq, labelVector, outFileNamePathStats)
        

    def calcStatClassifications(self,expLabel, notCorrectIndex):
        """
        Calculate statistics on classifications. How many were failed out of the total.  
        """   
        # Calculations for occurance statistics 
        # Get unique labels to a list present in expLabel
        labelVector = list(set(expLabel))
        # Loop through all labels present and count occurance of each label
        # Initiate array and loop. +1 is beacuse start of 0. Needed for indexing. 
        labelExpFreq = np.zeros(np.max(labelVector)+1)
        for currLabel in labelVector: 
            labelExpFreq[currLabel] = list(expLabel).count(currLabel)

        # Count the occurance of a label that is predicted to something else than expected
        # Init array and loop
        labelFailFreq = np.zeros(np.max(labelVector)+1)
        # For every item in notCorrectIndex
        for item in notCorrectIndex: 
            # Go though the label vector
            for currLabel in labelVector: 
                # If current label vector corresponds to the expected label value of the current notCorrectIndex
                if currLabel == expLabel[item]:      
                    # Add to frequency counter for current label
                    labelFailFreq[currLabel] += 1

        # Calculate percentage of number of failed objects with respect to number of available objects for each label
        # Round to two decimals
        labelFailPercentage = np.round((labelFailFreq / labelExpFreq * 100),2)
        # Return data
        return labelFailFreq, labelExpFreq, labelFailPercentage, labelVector