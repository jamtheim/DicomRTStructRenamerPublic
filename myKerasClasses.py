# Author: Christian Jamtheim Gustafsson, PhD

import os
import numpy as np
import sys
import nibabel as nib
from tensorflow import keras
# from tensorflow.keras.experimental import terminate_keras_multiprocessing_pools
import cv2
import matplotlib.pyplot as plt
from myPreProcessingClass import myPreProcessing
# import keras as kerasStandAlone
# from numpy.random import shuffle
# import threading

class DataGenerator(keras.utils.Sequence):
    """
    Creates a data generator which reads numpy arrays from HDF5 database
    The method before was based on reading files directly from the patient folders.
    Inputs:
        list_IDs (list): List with subject names
        labels (list): List with labels (for CAE, list_IDs = labels)
        data_dir: Directory of patient folders
        ObjectImageLabelDatabase: open file for iamge and label database read (alternative to reading from patient files)
        n_channels (int): number of image input channels
        dim (tuple): Dimensions of the input image (H,W,Depth)
        data_dir (str): Data directory of the folders
        shuffle (bool): If True, List_IDs get shuffled after each epoch
    Returns:
        None
    """
    def __init__(self, list_IDs, labels, data_dir, ObjectImageLabelDatabase, batch_size=32, dim=(512,512), n_channels=2, n_classes=13, shuffle=True):
        'Initialization'
        self.dim = dim
        self.data_dir = data_dir
        self.ObjectImageLabelDatabase = ObjectImageLabelDatabase
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        """
        Determines the number of batches (steps) per epoch.
        Inputs:
            None
        Returns:        
            batches (int): Number of batches (steps)
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        """
        Generates one batch of data.
        Inputs:
            index (int): determines the start index for the batch in the given training data
        Returns:
            X, y (array): returns data and corresponding labels (ground truth)
        """
        # with self.thread_lock:
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]       
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, labels_temp)
        return X, y
       

    def on_epoch_end(self):
        """
        Updates and shuffles indexes after each epoch.
        Inputs:
            None
        Return:
            None
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp, labels_temp):
        """
        Generates data containing batch_size samples
        Inputs:
            list_IDs_temp (list): list of ID indexes used to create on batch of data
            labels_temp: the corresponding labels
            However, this is changed to using a HDF5 database which contains all information on image data and label
        Returns:
            Data (array): Data used for training
        """
        # Initialization of arrays
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Latest method uses the HDF5 database, can thereby load 4 GPUs to almost 100% utlilization. 

            # Projection data with AddMap
            # Original way for running 109
            # Read Tra 2D projection slice
            X[i,:,:,0] = np.array(self.ObjectImageLabelDatabase[ID]['TraStructureProj'])
            # Read BodyAndOtherAdd2D transversal slice
            X[i,:,:,1] = np.array(self.ObjectImageLabelDatabase[ID]['AddMap'])
            # Read Cor 2D projection slice
            X[i,:,:,2] = np.array(self.ObjectImageLabelDatabase[ID]['CorStructureProj'])
            # Read Sag 2D projection slice
            X[i,:,:,3] = np.array(self.ObjectImageLabelDatabase[ID]['SagStructureProj'])

            # Projection data without AddMap
            # Read Tra 2D projection slice
            #X[i,:,:,0] = np.array(self.ObjectImageLabelDatabase[ID]['TraStructureProj'])
            # Read Cor 2D projection slice
            #X[i,:,:,1] = np.array(self.ObjectImageLabelDatabase[ID]['CorStructureProj'])
            # Read Sag 2D projection slice
            #X[i,:,:,2] = np.array(self.ObjectImageLabelDatabase[ID]['SagStructureProj'])

            # CoM data with AddMap
            # Read Tra 2D projection slice
            #X[i,:,:,0] = np.array(self.ObjectImageLabelDatabase[ID]['TraStructureCoM'])
            # Read BodyAndOtherAdd2D transversal slice
            #X[i,:,:,1] = np.array(self.ObjectImageLabelDatabase[ID]['AddMap'])
            # Read Cor 2D projection slice
            #X[i,:,:,2] = np.array(self.ObjectImageLabelDatabase[ID]['CorStructureCoM'])
            # Read Sag 2D projection slice
            #X[i,:,:,3] = np.array(self.ObjectImageLabelDatabase[ID]['SagStructureCoM'])

            # CoM data without AddMap
            # Read Tra 2D projection slice
            #X[i,:,:,0] = np.array(self.ObjectImageLabelDatabase[ID]['TraStructureCoM'])
            # Read Cor 2D projection slice
            #X[i,:,:,1] = np.array(self.ObjectImageLabelDatabase[ID]['CorStructureCoM'])
            # Read Sag 2D projection slice
            #X[i,:,:,2] = np.array(self.ObjectImageLabelDatabase[ID]['SagStructureCoM'])

            # Read label
            y[i] = int(np.array(self.ObjectImageLabelDatabase[ID]['Label']))
            # Used the input before from labels_temp as that list is inputted. 
            # y[i] = labels_temp[i] 
            # However, there is a 1:1 correspondance between the labels_temp and the ObjectImageLabelDatabase
            # They are both created from the ObjectLabelDatabase CSV file from the beginning
            # It felt more natural to read the label from the pre-created HDF5 database. 
            
            # Check data correctness here by plotting
            # imgplot = plt.imshow(X[2,:,:,0]); plt.show(block=False); plt.pause(5); plt.close()
            # imgplot = plt.imshow(X[2,:,:,1]); plt.show(block=False); plt.pause(5); plt.close()
            # imgplot = plt.imshow(img2D); plt.show(block=False); plt.pause(5); plt.close()
            # imgplot = plt.imshow(imgAllAdd2D); plt.show(block=False); plt.pause(5); plt.close()
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


        