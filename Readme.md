# Usage
The use this repository yourself you need following packages: torch, torchvision, opencv, numpy, argparse, BeautifulSoup, selenium, datetime, PIL, scipy, tqdm, prefetch_generator, torchsummary, optuna, 
## Inference
For inference of pictures you just need to start inference.py with the needed options, Picture Inference explains this in more detail. For Video Inference you just need to start data_capture.py.
## Training
To start training you first need a dataset. NH-Haze is recommended. First the Dataset needs to be structured correctly. To do this you need to start myutils.py change the code for your dataset. The Hyperparameters, Paths and other options can also be changed in params.py. After this is done you can start training by starting train.py
## Dataset Creation
For Dataset creation you first need to install firefox in the default location. It is also strongly recommended to download an addblocker for firefox as an xpi file. After this is done you can start data_crawler.py with the outpath as an option. The data_crawler opens firefox on your PC and screenshots the current screenoutput. To create your own dataset you need to start this programm, get your mouse out of frame and then you can not do anything on your PC or it will output the pictures currently on screen which might not be firefox. It is recommended to do this with an raspberrypi and to then just unplug the keyboard and mouse and to just let it run. The Outputted data is not processed it is only the screenshots of the current Screen.
# Dataset
data_crawler.py, geckodriver.exe are needed to create the CustomDehaze Dataset. Firefox in the default Windows install Location is also needed if you dont have firefox in the default location then you need to change the options.binary_location variable to the firefox location on your device. The data_crawler can be used with 
`python3 data_crawler.py --outpath "/path/to/Output/"`
an additional parameter --pathtoaddon can be set to the path to an adblock.xpi file to remove all ads on the website. --outpath gives the programm the path to the Output folder. The data crawler gets fogless, sligtly foggy, medium foggy, and strong foggy images from the website earthcams by screenshotting the the livecams. It only outputs the whole screenshot and no postprocessing is made.

# Picture Inference
inference.py is the file needed for picture inference. 
`python3 inference.py --modelpath "/path/to/.pth" --inferencepath "/path/to/hazypictures/" --gtpath "/path/to/groundtruth/" --gpu_mode True/False`
can be used to start inferencing all the pictures in the inferencepath. modelpath is the path to the .pth files of the model default is the folder ./weights with modelfile 999Dehaze.pth. gtpath is the path to the folder of the ground truth pictures. The ground truths have to be sorted the same way as the hazy files. If a gtpath is given then the ground truths will be outputted in the results directory and a PSNR score will be calculated and outputted. gpu_mode true enables if the GPU should be used for inference (8GB VRAM is needed). The output is always to folder ./results (this folder gets automatically created).

# Webcam and Video inference
data_capture is the file needed for Camera/Video inference.
`python3 data_capture.py --modelpath "/path/to/.pth"`
can be used to start camera inference. The modelpath is defaulty "./weights/999Dehaze.pth. The built-in camera will be defaulty used. If a different camera should be used, then the stream_id in CamStream needs to be set to the corresponding number this needs to be done in the code itself, i.e. built-in = 0, first = 1, second = 2 etc. A Video can also be dehazed by inputing the path to the video instead of the number. data_capture.py outputs a opencv video output with the dehazed Video/Camstream running. The Inferencetime is long which means that the outputted video will have few fps.

# Training
data.py and training.py are files that define classes that are used in train.py. loss.py defines the loss function. These Files should not need to be changed. train.py can not be changed via arguments and can only be changed by changing the hparams dictionary in params.py. Everything in train.py except for the training.train functions are definitions. The training.train function starts the train loop for n epochs for one dataset. train.py is the script that starts the training.

# Model
the model itself as well as most blocks are defined in models.py. basic_models.py also defines Blocks but since the Blocks that are defined in basic_models.py can also be useful in different Projects, they have been seperated from the other blocks. 

# Utility
Multiple scripts have been created to make everything easier. tune.py is a script that is used to tune the Hyperparameters.The params dictionary defines the Hyperparameters that will be tuned and also defines the range and specific numbers that will be chosen for a tuning loop. params.py defines all the Hyperparameters and Constants. parameter_print.py is useful to test, debug and analyze different Blocks and Models as well as to print the parameters of those. myutils.py is a collection of functions that are used in the project as well as a script to transform different datasets to a uniform datastructure you need this structure of data to be able to train the model without much modification.