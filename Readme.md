# Table of Content
## Dataset
data_crawler.py, geckodriver.exe are needed to create the CustomDehaze Dataset. Firefox in the default Windows install Location is also needed if you dont have firefox in the default location then you need to change the options.binary_location variable to the firefox location on your device. The data_crawler can be used with 
> python3 data_crawler.py --outpath "/path/to/Output/"
an additional parameter --pathtoaddon can be set to the path to an adblock.xpi file to remove all ads on the website.

## Picture Inference
inference.py is the file needed for picture inference. 
> python3 inference.py --modelpath "/path/to/.pth" --inferencepath "/path/to/hazypictures/" --gtpath "/path/to/groundtruth/" --gpu_mode True/False
can be used to start interferencing all the pictures in the inferencepath. modelpath is the path to the .pth files of the model default is the folder weights with modelfile 999Dehaze.pth. gtpath is the path to the folder of the ground truth pictures. The ground truths have to be sorted the same way as the hazy files. If a gtpath is given then the ground truths will be outputted in the results directory and a PSNR score will be calculated and outputted. gpu_mode true enables if the GPU should be used for inference (8GB VRAM are needed). 

## Webcam and Video inference
data_capture is the file needed for Cam inference.
> python3 data_capture.py --modelpath "/path/to/.pth"
can be used to start cam inference. The modelpath is defaulty "./weights/999Dehaze.pth. The built-in camera will be defaulty used. If a different camera should be used, then the stream_id in CamStream needs to be set to the corresponding number, i.e. built-in = 0, first = 1, second = 2 etc.

## Training
data.py and training.py are files that define classes that are used in train.py.loss.py defines the loss function. These Files should not need to be changed. train.py can not changed via arguments and can only be changed by changing the hparams dictionary in params.py. Everything in train.py except for the training.train functions are definitions. The training.train function starts the train loop for n epochs for one dataset.

## Model
the model itself as well as most blocks are defined in models.py. basic_models.py also defines Blocks but since the Blocks that are defined in basic_models.py can also be useful in different Projects, they have been seperated from the other blocks.

## Utility
Multiple scripts have been created to make everything easier. tune.py is a script that is used to tune the Hyperparameters.The params dictionary defines the Hyperparameters that will be tuned and also defines the range and specific numbers that will be chosen for a tuning loop. params.py defines all the Hyperparameters and Constants. parameter_print.py is useful to test, debug and analyze different Blocks and Models as well as to print the parameters of those. myutils.py is a collection of functions that are used in the project as well as a script to transform different datasets to a uniform datastructure you need this structure of data to be able to train the model without much modification.