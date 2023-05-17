# vowelsLSE Sign Recognition

In this work, we presented a sign recognition system based on hand detection and image classification of the vowels of the Spanish Sign language (Lengua de Signos Española - LSE), to make the recognition of the vowels of the LSE in real-time. The vowelsLSE dataset consists of 5 gestures of one person signing each vowel according to LSE - Lengua de Signos Española (Spanish Sign Language) and contains 3461 images. It consists of RGB images in JPG format with a size of 400 x 400 and has a white background to make them the same size. This dataset has been created as a proof of concept and is being worked on for improvement in future updates. 

The repository includes:

        inference_images folder contains images for making inferences once the classification models were trained.
        
        integration_codes folder includes four Python scripts where the detection and classification models get in merged to make the recognition of the vowels of the LSE:
        
          - DataCollection.py to collect and create the vowelsLSE dataset.
          
          - HandTrackingModule_noSkeleton.py is the hand detection module.
          
          - classificationModule_init.py is the classification module for the vowels of LSE.
          
          - signRecognition_init.py is a recognition module for the vowels of LSE in real-time using the hand detection and image classification modules.  
          
        models folder has two image classification models:    
        
          - NN2_keras.ipynb. It is a complex neural network created with Keras and Tensorflow libraries using Keras Tuner for hyperparameter search.
          
          - ResNet50_fastai.ipynb. It is a model which apply ResNet50 architecture and used the FastAI library. 
          
        notebook_images folder contains two images, one indicates the transformations applied for data, and the other is the signs of the vowels of the LSE.

The corresponding datasets for Keras and FastAI models are in the following link: https://unirioja-my.sharepoint.com/:f:/g/personal/maalvear_unirioja_es/EvNHeOc-orlLtrT67Y4-w-EBT8JWvD5aTwVImW6PmP4i5A?e=9SDtfF
