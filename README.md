# DicomRTStructRenamerPublic
Deep learning-based method for automated renaming and standardization of prostate DICOM radiotherapy structure annotations.

This repository contains code and documentation used in the publication "XXX" by Jamtheim Gustafsson et al. 2021 XXX. Please connect the numbers in the list below to the number in each file present in the repository folder. It represents the natural flow of code execution, from data preparation to model inference. 

(_2.) QA and integrity check of new data\
(_1.) Sorting of raw DICOM data\
(0.) Extraction of test dataset\
(0.5.) Anonymization of dataset \
(1.) Generation of Nifti files from DICOM raw data\
(2.) Creation of dataset and training database\
(2.5.) QA of radiotherapy structure geometry\
(3.) Model training\
(4.) Model inference

Contact: christian.JamtheimGustafsson@skane.se

Requirements:\
dcmrtstruct2nii==1.0.18\
h5py==2.10.0\
keract==4.3.4\
Keras==2.4.3\
Keras-Applications==1.0.8\
Keras-Preprocessing==1.1.2\
matplotlib==3.2.1\
nibabel==3.0.1\
numpy==1.18.5\
opencv-python==4.2.0.32\
pandas==1.0.1\
Pillow==7.1.2\
psutil==5.7.0\
pydicom==1.4.2\
PyYAML==5.3\
scikit-image==0.16.2\
scikit-learn==0.22.2.post1\
scipy==1.4.1\
SimpleITK==1.2.4\
tensorboard==2.2.2\
tensorboard-plugin-profile==2.2.0\
tensorboard-plugin-wit==1.7.0\
tensorflow-estimator==2.2.0\
tensorflow-gpu==2.2.0\
tf-keras-vis==0.5.5
