# Author: Christian Jamtheim Gustafsson, PhD

import os
import numpy as np
import sys
import nibabel as nib
import cv2
from PIL import Image

class myPreProcessing: 
    
    def __init__ (self): 
        pass

    def nii2nparray(self, niiFilePath):
        """
        Load nii data file and return Numpy array
        """
        # Read file with nii babel method
        # Sets orientation the same as original in ITK-Snap
        niimgdata = nib.load(niiFilePath)   
        # Convert data to numpy array 
        npdata = niimgdata.get_fdata()
        return niimgdata, npdata
   

    def loadAndResize(self, niiFilePath, dim1, dim2):
        """
        Load nii data file and change data dimensions through neirest neighbour interpolation
        Returns interpolated 2D image
        """
        # Read file with nii babel method
        # Sets orientation the same as original in ITK-Snap
        # Convert nii data to numpy array 
        niimgdata, np2Ddata = self.nii2nparray(niiFilePath)

        # The following makes a difference for speed
        #if img2Dnpdata.dtype == 'float64': 
        #    # Cast to float32 for speed
        #    img2Dnpdata = img2Dnpdata.astype('float32')

        # Decrease size and store it
        np2DdataOut = cv2.resize(np2Ddata,(dim1,dim2),0,0,interpolation = cv2.INTER_NEAREST)
        # Pillow tryout
        # Did not make it faster with ordinary pillow or the pillow-SIMD
        # img2DOut = np.array(Image.fromarray(img2Dnpdata).resize((dim1,dim2), Image.NEAREST))
        return np2DdataOut, niimgdata

