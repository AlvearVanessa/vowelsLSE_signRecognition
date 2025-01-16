# vowelsLSE Sign Recognition

In this work, we present a sign recognition system based on hand detection and image classification of the vowels of the Spanish Sign language (Lengua de Signos Española - LSE), to make the recognition of the vowels of the LSE in real-time. The vowelsLSE dataset consists of 5 gestures of one person signing each vowel according to LSE and contains 3461 images. It consists of RGB images in JPG format with a size of 400 x 400 and has a white background to make them the same size. This dataset has been created as a proof of concept and is being worked on for improvement in future updates. This work was based on the Hand Sign Detection for the American Sign Language (ASL) course on the following website: https://www.computervision.zone/courses/hand-sign-detection-asl/


The repository includes:

        inference_images folder contains images for making inferences once the classification models are trained.
        
        integration_codes folder includes six Python scripts where the detection and classification models get merged to make the recognition of the vowels of the LSE:
        
          - DataCollection.py to collect and create the vowelsLSE dataset.          
          - HandTrackingModule_noSkeleton.py is the hand detection module and works for Keras and FastAI libraries.          
          - classificationModule_init.py is the classification module for the vowels of LSE works for the Keras model.      
          - classificationModule_init_fastai.py is the classification module for the vowels of LSE works for the FastAI model.      
          - signRecognition_init.py is a recognition module for the vowels of LSE in real-time using the hand detection and image classification modules that work for the Keras model. 
          - signRecognition_init_fastai.py is a recognition module for the vowels of LSE in real-time using the hand detection and image classification modules that work for the FastAI model.
          
          
        The models folder has two image classification models:    
        
          - NN2_keras.ipynb. It is a complex neural network created with Keras and Tensorflow libraries using Keras Tuner for hyperparameter search.          
          - ResNet50_fastai.ipynb. It is a model that applies ResNet50 architecture and uses the FastAI library. 
          
        notebook_images folder contains two images, one indicates the transformations applied for data, and the other is the signs of the vowels of the LSE.

The corresponding datasets for Keras and FastAI models are in the following link: 
https://unirioja-my.sharepoint.com/:f:/g/personal/maalvear_unirioja_es/EvNHeOc-orlLtrT67Y4-w-EB5cMY1W69f3CXyAEC21rt7w?e=fnJ4KN
