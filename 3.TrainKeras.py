# File for initialize training
# Author: Christian Jamtheim Gustafsson, PhD

import numpy as np
import csv
import os
import cv2
import os.path
import time
import h5py
import subprocess
import nibabel as nib
import shutil
import datetime
import platform
from pathlib import Path
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model, multi_gpu_model
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard, CSVLogger
from myKerasClasses import DataGenerator
from myEvaluationClasses import myEvaluation
from commonConfig import loadModel, loadModelInputConf, addRegularization, loadStructuresOfInterest
from myCrossVal import generateCSVFiles
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sharedFunctions import checkIfProcessRunning, findProcessIdByName, getClassWeights, printClassWeights
from shutil import copyfile

def TrainKeras():

    ############################################
    # Config section
    ############################################
    # Experiment iteration number and dataset version
    organ = 'Prostate'
    useModel = 'InceptionResNetV2' 
    expIter = 130
    datasetVersion = 42
    epochs = 100
    useMixedPrecision = False # Note: Does not run validation process in this mode. 
    useBatchSize = 72 #72 for InceptionResNetV2
    nbrCrossVals = 10
    createNewCVFiles = 1
    useAugData = False
    useClassWeights = True
    # If only one run is needed (nbrCrossVals=1)
    valSplitSingleRun = 0.1
    # Repeat training if nbrCrossVals=1
    repeatRoundsTraining = 1
    # Parameters for number of GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    nrCUDADevicesUsed = sum(c.isdigit() for c in os.environ["CUDA_VISIBLE_DEVICES"])
    
    # Configure memory to grow as needed, do not pre allocate memory
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu in physical_devices: 
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set options for mixed precicion, can thereby double batch size
    if useMixedPrecision == True: 
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        mixedPrecisionBatchFactor = 2
    else:
        mixedPrecisionBatchFactor = 1
            
    # Load model input configuration
    modelInputConfig = loadModelInputConf(organ)
    # Set batch size
    useChannels = modelInputConfig.useChannels
    useClasses =  modelInputConfig.useClasses
    useResolution = modelInputConfig.useResolution
    useBatchSize = useBatchSize * mixedPrecisionBatchFactor * nrCUDADevicesUsed
    # Data paths
    ProjectPath = '/mnt/mdstore1/Christian/Projects/StructFinder'
    tensorboardLogDirBase = ProjectPath + '/reports'
    modelSaveDirBase = ProjectPath + '/models'
    dataSetInput = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + \
        str(datasetVersion) + '/' + 'AllData2D' + '_' + str(useResolution[0]) + '_' + str(useResolution[1])
    #Orig 100% of data
    ObjectLabelDatabaseFilePath = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + str(datasetVersion) + '/' + 'ObjectLabelDatabase.csv'
    #75% of data
    #ObjectLabelDatabaseFilePath = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + str(datasetVersion) + '/' + 'ObjectLabelDatabase_sorted_0.75.csv'  
    #50% of data
    #ObjectLabelDatabaseFilePath = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + str(datasetVersion) + '/' + 'ObjectLabelDatabase_sorted_0.50.csv'  
    #25% of data
    #ObjectLabelDatabaseFilePath = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + str(datasetVersion) + '/' + 'ObjectLabelDatabase_sorted_0.25.csv'  
    #10% of data
    #ObjectLabelDatabaseFilePath = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + str(datasetVersion) + '/' + 'ObjectLabelDatabase_sorted_0.10.csv'  
    ObjectImageLabelDatabaseFilePath = ProjectPath + '/' + 'dataset/Final/StructFinderDataset' + organ + 'V' + str(datasetVersion) + '/' + 'ObjectImageLabelDatabase.hdf5'
    dataSetSplitOutput = ProjectPath + '/' + 'src/models/DataSetSplits' + '/' + 'StructFinder_exp' + str(expIter)

    # Create CSV files from the dataSet contents 
    if createNewCVFiles: 
        generateCSVFiles(dataSetInput, ObjectLabelDatabaseFilePath, dataSetSplitOutput, nbrCrossVals, useAugData, valSplitSingleRun, repeatRoundsTraining)

    # Set training and validation parameters
    TrainParams = {'dim': useResolution,
                   'batch_size': useBatchSize,
                   'n_classes': useClasses,
                   'n_channels': useChannels,
                   'shuffle': True}
    ValParams = {'dim': useResolution,
                    'batch_size': useBatchSize,
                    'n_classes': useClasses,
                    'n_channels': useChannels,
                    'shuffle': True}

    # To get all data points in inference for evaluation 
    # of failed subjects use batch_size 1
    # We do this to avoid missing some elements 
    # that is not even dividable with batch size
    ValParamsPredict = {'dim': useResolution,
                    'batch_size': 1,
                    'n_classes': useClasses,
                    'n_channels': useChannels,
                    'shuffle': False}

    # Set model name
    model_name = 'StructFinder_exp' + str(expIter)
    # Check if tensorboard is running and kill the session  
    if checkIfProcessRunning('tensorboard'):
        # Get list of pids
        listOfProcessIds = findProcessIdByName('tensorboard')
        # Get the pid   
        if len(listOfProcessIds) == 1: 
            killPid = listOfProcessIds[0]['pid']
            try:
                os.kill(killPid,9)
            except:
                print('Could not kill tensorboard (wrong user?)')

    # Start tensorflow and firefox
    tmpLogDir = tensorboardLogDirBase + '/' + model_name
    tensorBoardCmd = "tensorboard --port=6008 --logdir " + tmpLogDir
    webBrowseCmd = "firefox http://localhost:6008"
    tensorBoard = subprocess.Popen(
        [tensorBoardCmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
    webBrowse = subprocess.Popen(
        [webBrowseCmd], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)

    # Check how many split files were produced 
    nrCVFiles = len(os.listdir(dataSetSplitOutput))
    # Loop over all cross validations or repeats
    # Variable is named crossVal even for repeats
    for crossVal in range(1, nrCVFiles+1):
        # Define model name for the cross validation
        print('Starting cross validation ' + str(crossVal))
        # Reset data containers
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        # Read CSV file for the current cross validation
        # Create vectors for data pointers and correspodning label
        csv_file = dataSetSplitOutput + '/' + \
            "dataset_split_cross_" + str(crossVal) + '.csv'
        with open(csv_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if row[1] == 'training':
                    X_train.append(row[0])
                    y_train.append(row[2])
                elif row[1] == 'validation':
                    X_val.append(row[0])
                    y_val.append(row[2])


        # Initialize the HDF5 ObjectImageLabel database file for read only
        # This is used to feed the network with data 
        ObjectImageLabelDatabase = h5py.File(ObjectImageLabelDatabaseFilePath, 'r')

        # Data generators (both dataSetInput and ObjectImageLabelDatabase)
        training_generator = DataGenerator(
            X_train, y_train, dataSetInput, ObjectImageLabelDatabase, **TrainParams)
        validation_generator = DataGenerator(
            X_val, y_val, dataSetInput, ObjectImageLabelDatabase, **ValParams)
        validation_generator_predict = DataGenerator(
            X_val, y_val, dataSetInput, ObjectImageLabelDatabase, **ValParamsPredict)     

        # Callbacks
        # Tensorboard, dir converted to OS format
        log_dir = tensorboardLogDirBase + '/' + model_name + '/cv' + \
            str(crossVal) + '_' + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path(log_dir)
        # Define Tensorboard callback with profiling a batch
        tensorboard_callback = TensorBoard(
            log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, profile_batch=0) #Turning off profiler with profile_batch=0
        # Save model checkpoint iteration
        modelSaveDir = modelSaveDirBase + '/' + \
            model_name + '/cv' + str(crossVal)
        os.makedirs(modelSaveDir, exist_ok=True)
        # Save model for each epoch
        checkpoint_callback = ModelCheckpoint(modelSaveDir + '/' + "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                              monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')
        # If we are running in a mode where the whole dataset is used and no validation
        if valSplitSingleRun == 0.0:
            checkpoint_callback = ModelCheckpoint(modelSaveDir + '/' + "weights.{epoch:02d}.hdf5",
                                              verbose=1, save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch')

        # CSV logger
        CSV_callback = CSVLogger(
            modelSaveDir + '/logEpoch.csv', separator=',', append=True)
        
        reduce_lr = ReduceLROnPlateau(
                 monitor="val_accuracy",
                 factor=0.2,
                 patience=10,
                 verbose=1,
                 mode="max",
                 min_delta=0.00001,
                 cooldown=0,
                 min_lr=0)

        # Early stopping        
        early_stop = EarlyStopping(
                monitor="val_accuracy",
                min_delta=0,
                patience=15,
                verbose=1,
                mode="max",
                baseline=0,
                restore_best_weights=False)
    
                
        # Collect callbacks to use
        CallbackList = [tensorboard_callback,
                        checkpoint_callback,
                        CSV_callback,
                        reduce_lr,
                        # early_stop
                        ]

        # Copy this file as config log
        configLogFileName = modelSaveDirBase + '/' + model_name + '/' + 'runCode_StructFinder_exp' + str(expIter) + '.py'
        copyfile(__file__, configLogFileName)

        # Get class weights (None or calculated)
        myClassWeights = getClassWeights(useClassWeights, useClasses, y_train)
        # Print the class weights to a archive file   
        classWeightsFileName = modelSaveDirBase + '/' + model_name + '/' + 'classWeights_StructFinder_exp' + str(expIter) + '.txt'
        printClassWeights(myClassWeights, classWeightsFileName)

        # Create a MirroredStrategy for parallell GPUs 
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            # Load model for training and do not init any weights (random)
            model = loadModel('training', None, organ, useModel)
            # Add regularization
            # addRegularization(model, tf.keras.regularizers.l2(0.001))
            
        # Train model on dataset
        # Workers = 1, max_queue_size=100 good for 1 or 2 GPUs using 2 channels
        # For Multip GPUs, workers=2 gives ValueError: task_done(), avoid it
        model.fit(training_generator,
                   validation_data=validation_generator,
                   verbose=1,
                   epochs=epochs,
                   workers=1, 
                   max_queue_size=50,
                   use_multiprocessing=False,
                   callbacks=CallbackList,
                   class_weight=myClassWeights)

        # Write out failing objects and statistics from validation dataset to a file
        # For valSplitSingleRun = 0, there is no validation data
        if valSplitSingleRun != 0 or nbrCrossVals > 1: 
            # Init class
            eval = myEvaluation()
            # Set file name for failed objects
            failedClassificationsFileName = 'failedClassifications_' + 'cv' + str(crossVal) + '_MaxEpoch' + str(epochs) + '.csv'
            # Set file name for object classification statistics 
            statClassificationsFileName = 'statClassifications_' + 'cv' + str(crossVal) + '_MaxEpoch' + str(epochs) + '.csv'
            # Load structures used, this is only to resolve the name after evaluation
            InferenceStructureNames = loadStructuresOfInterest(organ,'inference') 
            # Write out data for failed classifications
            eval.writeFailedClassifications(model,validation_generator_predict, modelSaveDir, failedClassificationsFileName, statClassificationsFileName, InferenceStructureNames)
       

# To solve potential problems with multiprocessing call this
if __name__ == '__main__':
    TrainKeras()
