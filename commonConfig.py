# Author: Christian Jamtheim Gustafsson, PhD

from tensorflow.keras.applications import ResNet50V2, ResNet101, Xception, ResNet101V2, ResNet152V2, InceptionResNetV2, NASNetLarge
#, EfficientNetB4 #TF2.3.0
import os
import tempfile
import tensorflow as tf
# For SEResNets
import se_resnet as AllSEResNets # Load local copy because this version is adapted, see comments in it. Can it be trusted? 
# import se_inception_resnet_v2 as AllSEInceptionResNets
# https://github.com/titu1994/keras-squeeze-excite-network


def getSignalTruncTh(case): 
    """
    Set threshold value for largest signal in a linear combination of structure masks
    """

    if case == 'BodyAndOtherAddMap':
        #BodyAndOtherAddMap was body + all other structs
        #Possible structure overlay is possible, therefore truncate the signal.
        signalTruncTh = 1

    # Return set value
    return signalTruncTh


def loadIgnoreStructures(organ): 
    """
    Output a list of structures to ignore or remove
    """
    if organ == 'Prostate': 
        # Create list of structures to ignore
        ignoreStructs = ['tuning', 'HELP', 'X','Y', 'Z', 'Dose', 'Match']
        # Convert to lower case to resolve problems in Linux 
        # as some organ names are written with capitals and some not.
        # This is not an issue in Windows as it is not caps sensitive. 
        ignoreStructs = [each_string.lower() for each_string in ignoreStructs]
        
    return ignoreStructs


def loadStructuresOfInterest(organ, mode): 
    """
    For training mode
    Output a list of structures of interest and a label vector
    By putting the struct in structs or in the special structs we can get different behavior when
    selecting source data, see getFileNamePathForStructure
    When adding structures, add label in AllLabelsOfInterest and in the structure list. Make sure adaptions are made in exclusion list of labels change. 
    For inference mode
    Output a name vector that corresponds to the label class number (add also here if adding class)
    """

    def defineLabelVector(): 
        """ 
        This allows naming to be customized for different input structures
        Please observe that order of vector elements is important here
        """
        AllLabelsOfInterest = [0,1,2,3,4,5,6,7,8,9,10,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,13,13,13,13,13,13,14,14,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,18,19,20,20,21,21]
        return AllLabelsOfInterest

    if organ == 'Prostate': 
        if mode == 'training': 
            # Create list of structures of interest to loop over
            # See word list for structure names
            structs = [
            'BODY', 'Bladder', 'FemoralHead_L','FemoralHead_R', # 0-3, 1
            'Rectum', 'Genitalia', 'Penilebulb', 'CouchSurface',           # 4-7, 1
            'CouchInterior', 'BowelBag', 'AnalCanal',                      # 8-10, 1
            'CTVT1_780', 'CTVT1_427', 'CTVT_427', 'CTVT1_500', 'CTVT_360', 'CTVT1_510', 'CTVT1_500pros', 'CTVT1_450',  # Prostate gland CTV, 11, 8 (label and number of elements)
            'PTVT1_780', 'PTVT1_427', 'PTVT_427', 'PTVT1_500', 'PTVT_360', 'PTVT1_510', 'PTVT1_500pros', 'PTVT1_450',  # Prostate gland PTV, 12, 8
            'CTVT2_700', 'CTVT2_500', 'CTVT3_500', 'CTVT2_700ves', 'CTVT2_500ves', 'CTVT3_500ves',      # Vesicle CTV 13, 6
            'PTVT2_700', 'PTVT2_500', 'PTVT3_500', 'PTVT2_700ves', 'PTVT2_500ves', 'PTVT3_500ves',      # Vesicle PTV 14, 6
            'GTVN1_600', 'GTVN2_600', 'GTVN3_600', 'GTVN1_640',                                         # Illiac lymphnode GTV 15, 4
            'CTVN1_600', 'CTVN2_600', 'CTVN3_600', 'CTVN1_640',                                         # Illiac lymphnode CTV 16, 4
            'PTVN1_600', 'PTVN2_600', 'PTVN3_600', 'PTVN1_640',                                         # Illiac lymphnode PTV 17, 4
            'CTVN_500',                                                                                 # Illiac elective other CTV 18, 1
            'PTVN_500',                                                                                 # Illiac elective other PTV 19, 1
            'CTVT1_600', 'CTVT1_700',                                                                   # Prostate + vesicle bed CTV 20, 2
            'PTVT1_600', 'PTVT1_700']                                                                   # Prostate + vesicle bed PTV 21, 2

            # Prostate gland boost volumes are defined as prostate gland, marked in correction list with prostate gland CTV and PTV classes. GTVT_78 existed (not used). 
            # CTVT1_700 with or without vesicles are classified as Bed.

            # Special Structures
            specialStructs = []
            
            # Convert to lower case to resolve problems in Linux 
            # as some organ names are written with capitals and some not.
            # This is not an issue in Windows as it is not caps sensitive. 
            structs = [each_string.lower() for each_string in structs]
            specialStructs = [each_string.lower() for each_string in specialStructs]
            # Combine the two
            AllStructuresOfInterest =  structs + specialStructs
            # Define a label vector for these structures (order is important)
            # In this way we can group multiple structures to the same label
            AllLabelsOfInterest = defineLabelVector()
            
            # Check that label vector contains same number of elements as AllStructuresOfInterest
            if len(AllStructuresOfInterest) != len(AllLabelsOfInterest):
                raise ValueError('Not the same amount of elements in structure list as in label vector')
            return AllStructuresOfInterest, structs, specialStructs, AllLabelsOfInterest

        if mode == 'inference': 
            # Define the new names. This vector must be in correspondance to the label vector and contain the unique elements.
            # Tried to make the list in a smart way. 
            InferenceStructureNames = [
            'BODY', 'Bladder', 'FemoralHead_L','FemoralHead_R', 
            'Rectum', 'Genitalia', 'PenileBulb', 'CouchSurface', 
            'CouchInterior', 'BowelBag', 'AnalCanal', 
            'GlandCTV',
            'GlandPTV',
            'VesicleCTV',
            'VesiclePTV',
            'IlLymGTV',
            'IlLymCTV',
            'IlLymPTV',
            'IlElecCTV',
            'IlElecPTV',
            'BedCTV',
            'BedPTV']         
            
            # Get label vector
            AllLabelsOfInterest = defineLabelVector()
            # Check that label vector contains same number of unique elements as InferenceStructureNames
            if len(InferenceStructureNames) != len(set(AllLabelsOfInterest)):
                raise ValueError('Not the same amount of unique elements in inference structure list as unique values in label vector')

        return InferenceStructureNames


def addRegularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):
    """
    Add regularization to the model
    """
    # See https://sthalles.github.io/keras-regularizer/
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()
    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)
    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    print('Regularization added')
    return model


def loadModelInputConf(organ): 
    """
    Load input configuration for the data to the model
    """
    if organ == 'Prostate': 
        class setConfig: 
            pass
        # Get all structures and labels
        AllStructures, structs, specialStructs, AllLabelsOfInterest = loadStructuresOfInterest(organ,'training')
        # Define config
        config = setConfig()
        # Define configuration
        config.useChannels = 4
        # Original for running iter 109
        # config.useChannels = 4
        config.useClasses = len(set(AllLabelsOfInterest))
        config.useResolution = (256,256)
        # config.useResolution = (299,299)

    return config


def loadModel(mode, modelWeights, organ, modelType): 
    """
    Load model and compile it 
    Input training or inference mode, model weights and type of model 
    Return model
    """
    # Load model input configuration
    modelInputConfig = loadModelInputConf(organ)
    # Get values 
    useChannels = modelInputConfig.useChannels
    useClasses =  modelInputConfig.useClasses
    useResolution = modelInputConfig.useResolution

     # Define model
    if modelType == 'ResNet101': 
        model = ResNet101(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses)
    elif modelType == 'SEResNet101':    
            mySEResNet = AllSEResNets.SEResNet101
            model = mySEResNet(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses)
    elif modelType == 'SEResNet154':    
            mySEResNet = AllSEResNets.SEResNet154
            model = mySEResNet(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses)
    # elif modelType == 'SEInceptionResNetV2':    
    #         mySEInceptionResNet = AllSEInceptionResNets.SEInceptionResNetV2
    #         model = mySEInceptionResNet(include_top=True, weights=modelWeights, input_shape=(
    #             useResolution[0], useResolution[1], useChannels), classes=useClasses)
    elif modelType == 'EfficientNetB4':    
            model = EfficientNetB4(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses, classifier_activation="softmax")
    elif modelType == 'Xception':    
            model = Xception(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses)
    elif modelType == 'ResNet101V2':    
            model = ResNet101V2(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses, classifier_activation="softmax")
    elif modelType == 'ResNet152V2':    
            model = ResNet152V2(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses, classifier_activation="softmax")
    elif modelType == 'InceptionResNetV2':    
            model = InceptionResNetV2(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses, classifier_activation="softmax")
    elif modelType == 'ResNet50V2':    
            model = ResNet50V2(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses, classifier_activation="softmax")
    elif modelType == 'NASNetLarge':    
            model = NASNetLarge(include_top=True, weights=modelWeights, input_shape=(
                useResolution[0], useResolution[1], useChannels), classes=useClasses)

    else: 
        raise ValueError('The selected model could not be found')
                 
    if mode == 'training': 
        print('Loaded model ' + modelType + ' for training, no weights loaded')
        # Add reglizarization if needed
        # model = addRegularization(model, tf.keras.regularizers.l2(0.0000))
    if mode == 'inference': 
        print('Loaded model ' + modelType + ' for inference, weights loaded.')
        # Do not add regularization 

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    # metrics=['accuracy']
                    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
                    weighted_metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
                    )

    return model


