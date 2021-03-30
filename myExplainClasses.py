# Author: Christian Jamtheim Gustafsson, PhD

import os
import numpy as np

class myExplainModels:

    def __init__ (self): 
        pass

    def rotateMatrixForDislay(self, matrixIn, case): 
        """
        Rotate data in matrix to display them in a better view
        Input data is represented by the 4 image channels, 
        either as image data or a Saliency map with 4 channels. 
        """
        import cv2
        import numpy as np

        if case == 'ChannelWise': 
            # print(matrixIn.shape)
            assert matrixIn.shape == (256, 256, 4)
            # Create zero valued matrix output 
            matrixOut = np.zeros(matrixIn.shape)
            # Operate on each image channel to rotate and flip the data individually
            matrixOut[:,:,0] = cv2.flip(cv2.rotate(matrixIn[:,:,0], cv2.ROTATE_90_CLOCKWISE),1)
            matrixOut[:,:,1] = cv2.flip(cv2.rotate(matrixIn[:,:,1], cv2.ROTATE_90_CLOCKWISE),1)
            matrixOut[:,:,2] = cv2.rotate(matrixIn[:,:,2], cv2.ROTATE_90_COUNTERCLOCKWISE)
            matrixOut[:,:,3] = cv2.rotate(matrixIn[:,:,3], cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Return data rectum
        return matrixOut



    def GradCamGuided(self, X, model, layerOfInterest):
        """
        This function generate guided grad cam saliency maps which is defined per input channel. 
        https://github.com/ismailuddin/gradcam-tensorflow-2/blob/master/notebooks/GradCam.ipynb
        """
        import tensorflow as tf
        import random
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        # Define input image 
        image = X[0,:,:,:]
        # Define model to use
        model = model

        @tf.custom_gradient
        def guided_relu(x):
            def grad(dy):
                return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
            return tf.nn.relu(x), grad


        class GuidedBackprop:
            def __init__(self, model, layer_name: str):
                self.model = model
                self.layer_name = layer_name
                self.gb_model = self.build_guided_model()

            def build_guided_model(self):
                gb_model = tf.keras.Model(
                    self.model.inputs, self.model.get_layer(self.layer_name).output
                )
                layers = [
                    layer for layer in gb_model.layers[1:] if hasattr(layer, "activation")
                ]
                for layer in layers:
                    if layer.activation == tf.keras.activations.relu:
                        layer.activation = guided_relu
                return gb_model

            def guided_backprop(self, image: np.ndarray):
                with tf.GradientTape() as tape:
                    inputs = tf.cast(image, tf.float32)
                    tape.watch(inputs)
                    outputs = self.gb_model(inputs)
                grads = tape.gradient(outputs, inputs)[0]
                return grads

        # We get the output of the last convolution layer. We then create a model that goes up to only that layer.
        last_conv_layer = model.get_layer(layerOfInterest)
        last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
        # We create a model which then takes the output of the model above, and uses the remaining layers to get the final predictions.
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer_name in ["avg_pool", "predictions"]:
            x = model.get_layer(layer_name)(x)
        classifier_model = tf.keras.Model(classifier_input, x)
        # First, we get the output from the model up till the last convolution layer. We ask tf to watch this tensor output, 
        # as we want to calculate the gradients of the predictions of our target class wrt to 
        # the output of this model (last convolution layer model).
        with tf.GradientTape() as tape:
            last_conv_layer_output = last_conv_layer_model(image[np.newaxis, ...])
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # The partial derivative / gradient of the model output (logits / prior to softmax), with respect to the feature map (filter) activations 
        # of a specified convolution layer (the last convolution layer in this case) is: 
        grads = tape.gradient(top_class_channel, last_conv_layer_output)[0]
        last_conv_layer_output = last_conv_layer_output[0]
        # Guided backpropagation implementation:
        guided_grads = (tf.cast(last_conv_layer_output > 0, "float32") * tf.cast(grads > 0, "float32") * grads)
        pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1))
        guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)

        for i, w in enumerate(pooled_guided_grads):
            guided_gradcam += w * last_conv_layer_output[:, :, i]

        # Resize to image size
        guided_gradcam = cv2.resize(guided_gradcam.numpy(), (image.shape[0], image.shape[1]))

        gb = GuidedBackprop(model, layerOfInterest)
        saliency_map = gb.guided_backprop(image[np.newaxis, ...]).numpy()
        saliency_map = saliency_map * np.repeat(guided_gradcam[..., np.newaxis], image.shape[2], axis=2) #4 dimensions of input
 
        # Below aims to remove outliers so we can get reasable scaling of our data 
        # Subtract the mean 
        saliency_map -= saliency_map.mean()
        # Divide by std get Z-score, zero mean and 1 in std. 
        saliency_map /= saliency_map.std() + tf.keras.backend.epsilon()
        # Multiply by a number which will be the new std. 
        saliency_map *= 0.25
        # Lets go for a pm 2 std for value coverage. This is why 0.25 was chosen above. 
        # Adding 0.5 will lift everything to positive,
        # as range right now is -0.5 to 0.5. 
        saliency_map += 0.5
        # Clip it to [0,1]. This means we got pm 2 std of the data around the mean converted to [0,1]. 
        saliency_map = np.clip(saliency_map, 0, 1)
        # If 256 range is what we want
        #saliency_map *= (2 ** 8) - 1
        #saliency_map = saliency_map.astype(np.uint8)

        # Return data
        return saliency_map


    def Activation_Keract(self, patientName, X, AllModels, ROINameCollected): 
        """
        Generate activation maps for each layer
        patientName, Input data, AllModels, ROIOfInterest
        Output: Keract maps saved in folders
        """  
        from keract import get_activations, display_activations, display_heatmaps, display_gradients_of_trainable_weights, get_gradients_of_activations, get_gradients_of_trainable_weights
        # For all models
        for i in range(len(AllModels)):
            # Make sure folders exist
            activationDir = './ActivationMaps' + '/' + patientName + '/' + ROINameCollected + '/' + 'model_' + str(i) + '/' + 'Activation'
            if not os.path.exists(activationDir):
                os.makedirs(activationDir)
            overlayDir = './ActivationMaps' + '/' + patientName + '/' + ROINameCollected + '/' + 'model_' + str(i) + '/' + 'Overlay'
            if not os.path.exists(overlayDir):
                os.makedirs(overlayDir)
            # Get activations
            currActivations = get_activations(AllModels[0], X, layer_names=None, nodes_to_evaluate=None, output_format='full', nested=False, auto_compile=True)
            # Save activations only
            display_activations(currActivations, cmap=None, save=True, directory=activationDir, data_format='channels_last', fig_size=(24, 24), reshape_1d_layers=False)
            # Save acivation overlay
            display_heatmaps(currActivations, X, save=True, directory=overlayDir)



    def Explain_tfkerasvis(self, X, model, mostCommonIndex):
        """
        Generate gradCAM plusplus images, Saliency Maps and Score cam images and save them.  
        Seems to work very well, but it does not generate channel specific saliency maps. 
        """
        from matplotlib import pyplot as plt
        from matplotlib import cm
        import tensorflow as tf
        from tf_keras_vis.utils import num_of_gpus, normalize
        from tf_keras_vis.gradcam import GradcamPlusPlus, Gradcam
        from tf_keras_vis.saliency import Saliency
        from tf_keras_vis.scorecam import ScoreCAM

        # The `output` variable refer to the output of the model,
        # so, in this case, `output` shape is ` (samples, classes).
        def loss(output):
            # mostCommonIndex is the index determined from the inference. 
            # This must be in correspondance with the true label, or else the map will be wrong. 
            return (output[0][mostCommonIndex])

        def model_modifier(model):
            model.layers[-1].activation = tf.keras.activations.linear
            return model

        # Make prediction 
        output = model.predict(X) 
        # Create GradcamPlusPlus object
        gradcamplusplus = GradcamPlusPlus(model,model_modifier=model_modifier, clone=False)
        # Generate heatmap with GradCAM and normalize it
        camplusplus = gradcamplusplus(loss, X, penultimate_layer=-1) # model.layers number
        camplusplus = normalize(camplusplus)
        # Create heat map with 4 channels
        heatmap_GradCamPlusPlus = np.uint8(cm.jet(camplusplus[0,:,:])[..., :3] * 255)


        # Create Saliency object.
        saliency = Saliency(model, model_modifier=model_modifier, clone=False)
        # Generate saliency map with smoothing that reduce noise by adding noise
        saliency_map = saliency(loss,
                                X,
                                smooth_samples=20, # The number of calculating gradients iterations.
                                smooth_noise=0.20) # noise spread level.
        # Normalize map and get rid of 1 for the batch size
        saliency_map = normalize(saliency_map)
        saliency_map = saliency_map[0,:,:]

        # Create ScoreCAM object
        scorecam = ScoreCAM(model, model_modifier, clone=False)
        # This cell takes a lot of time on CPU, prefer GPU but doable on CPU.
        # Generate heatmap with ScoreCAM
        cam_score = scorecam(loss,
                    X,
                    penultimate_layer=-1, # model.layers number
                    )
        cam_score = normalize(cam_score)
        # Create heatmap
        heatmap_CamScore = np.uint8(cm.jet(cam_score[0,:,:])[..., :3] * 255)
        # Return data
        return heatmap_GradCamPlusPlus, saliency_map, heatmap_CamScore


    def CreateExplainMaps(self, patientName, X, ROINameCollected, AllModels, mostCommonIndex):
        """
        Calulates and creates different explainatory maps, using the other defined functions in this class, and saves them into image overlay maps. 
        """
        import matplotlib.pyplot as plt

        # Define name of last convolutional layer (dependent on model)
        layerOfInterest = "conv_7b_ac"

        # Make sure folders exist for saving data
        GradCamDir = './Explain' + '/' + patientName + '/' + ROINameCollected +  '/' + 'GradCams'
        if not os.path.exists(GradCamDir):
            os.makedirs(GradCamDir)
        SaliencyMapDir = './Explain' + '/' + patientName + '/' + ROINameCollected + '/' + 'SaliencyMaps' 
        if not os.path.exists(SaliencyMapDir):
            os.makedirs(SaliencyMapDir)
        ExplainAllDir = './Explain' + '/' + patientName + '/' + ROINameCollected 
        if not os.path.exists(ExplainAllDir):
            os.makedirs(ExplainAllDir)

        # For each model 
        for i in range(0,len(AllModels)): 
            # Define model
            model = AllModels[i]

            # Get heat maps, saliency maps and CamScore (a bit slow)
            [HeatmapGradCamPlusPlus, SaliencyMap, CamScore] = self.Explain_tfkerasvis(X, model, mostCommonIndex)
            # Get channel specific Saliency map for the specific model 
            SaliencyMapChannelWise = self.GradCamGuided(X, model, layerOfInterest)
            # Extract the image (batch size =1)
            image = X[0,:,:,:]

            # For sake of easier viewing rotate the image and maps for the ChannelWise data
            # Did not do this for the rest of the data as it was not channel independent data. 
            SaliencyMapChannelWise_rot = self.rotateMatrixForDislay(SaliencyMapChannelWise, 'ChannelWise')
            image_rot = self.rotateMatrixForDislay(X[0,:,:,:],'ChannelWise')
            # Create a binary version of the image matrix, as AddMap is not binary. 
            image_rot_binary = np.zeros(image_rot.shape)
            image_rot_binary[image_rot>0] = 1

            # Plot maps
            subplot_args = { 'nrows': 1, 'ncols': image.shape[2], 'figsize': (16, 4),
                        'subplot_kw': {'xticks': [], 'yticks': []} }
            f,ax = plt.subplots(**subplot_args)
  
            # Overlay image for each channel with masked version of channel specific saliency map
            # Use loop for less code
            # for j in range(0,image.shape[2]): 
            """    
            # GradCamPlusPlus
            ax[0,j].imshow(image[:,:,j],cmap='gray')
            ax[0,j].imshow(HeatmapGradCamPlusPlus, cmap='jet', alpha=0.5)
            # Saliency map
            ax[1,j].imshow(image[:,:,j],cmap='gray')
            ax[1,j].imshow(SaliencyMap, cmap='jet', alpha=0.5)
            # Cam Score heatmaps
            ax[2,j].imshow(image[:,:,j],cmap='gray')
            ax[2,j].imshow(CamScore, cmap='jet', alpha=0.5)
            # Saliency map channel wise 
            ax[3,j].imshow(image_rot[:,:,j],cmap='gray')
            ax[3,j].imshow(SaliencyMapChannelWise_rot[:,:,j]*image_rot[:,:,j], cmap='jet', alpha=0.5) # Masked version
            """
            # Saliency map channel wise displayed in other order to match paper figure. Saliency mask are masked with structure (created binary). 
            ax[0].imshow(image_rot[:,:,0],cmap='gray')
            im0 = ax[0].imshow(SaliencyMapChannelWise_rot[:,:,0]*image_rot_binary[:,:,0], cmap='jet', alpha=0.5, vmin=0, vmax=1) # Masked version
            ax[1].imshow(image_rot[:,:,2],cmap='gray')
            im1 = ax[1].imshow(SaliencyMapChannelWise_rot[:,:,2]*image_rot_binary[:,:,2], cmap='jet', alpha=0.5, vmin=0, vmax=1) # Masked version
            ax[2].imshow(image_rot[:,:,3],cmap='gray')
            im2 = ax[2].imshow(SaliencyMapChannelWise_rot[:,:,3]*image_rot_binary[:,:,3], cmap='jet', alpha=0.5, vmin=0, vmax=1) # Masked version
            ax[3].imshow(image_rot[:,:,1],cmap='gray')
            im3 = ax[3].imshow(SaliencyMapChannelWise_rot[:,:,1]*image_rot_binary[:,:,1], cmap='jet', alpha=0.5, vmin=0, vmax=1) # Masked version
            
            # Tight the layout
            plt.tight_layout()
            # Add colorbar, all data have the same range of values so set hard coded range, vmin ,vmax, and use im0. 
            # This probably only work if all images have the same range, or else scale will not be representative. 
            f.colorbar (im0, ax=ax.ravel().tolist(), shrink=0.86, pad=0.01)
            
            # plt.show() # disable for correctly saving file
            plt.savefig(ExplainAllDir + '/' + 'ExplainMaps_' + 'model_' + str(i) + '.png')
            plt.close()
