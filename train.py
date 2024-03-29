from data import ImageDataset
from models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# my imports
import myutils
from loss import ChabonnierLoss, ColorLoss, GrayscaleLoss
from params import hparams
from training import Train

if __name__ == '__main__':

    np.random.seed(hparams["seed"])    # set seed to default 123 or opt
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])
    gpus_list = range(hparams["gpus"])
    cuda = hparams["gpu_mode"]
    hostname = str(socket.gethostname)
    cudnn.benchmark = True
    print(hparams)
    

    size = (hparams["height"], hparams["width"])

    print('==> Loading Datasets')
    #dataloader_cropo_haze = DataLoader(ImageDataset(hparams["train_data_path"] + "crop_O-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_cropnh_haze = DataLoader(ImageDataset(hparams["train_data_path"] + "crop_NH-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_cityscapes = DataLoader(ImageDataset(hparams["train_data_path"] + "cityscapes",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_nh_haze = DataLoader(ImageDataset(hparams["train_data_path"] + "NH-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    dataloader_o_haze = DataLoader(ImageDataset(hparams["train_data_path"]+ "O-Haze",size,hparams["crop_size"],hparams["scale_factor"],hparams["augment_data"]), batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["threads"])
    
    # define the Network
    Net = Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"],hparams["down_deep"] ,hparams["gpu_mode"], hparams["scale_factor"], hparams["pseudo_alpha"], hparams["hazy_alpha"], size=(hparams["height"], hparams["width"]))
    #Net = CarDetection(3,64,64,6,size=(hparams["width"],hparams["height"]))
    # print Network parameters
    pytorch_params = sum(p.numel() for p in Net.parameters())
    print("Network parameters: {}".format(pytorch_params))

    # set criterions & optimizers
    criterion = ChabonnierLoss(eps = 1e-6)
    optimizer = optim.Adam(Net.parameters(), lr=hparams["lr"], betas=(hparams["beta1"],hparams["beta2"]))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 250, 0.6)

    cuda = hparams["gpu_mode"]
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")


    torch.manual_seed(hparams["seed"])
    if cuda:
        torch.cuda.manual_seed(hparams["seed"])

    if cuda:
        Net = Net.to(torch.device("cuda:0"))
        criterion = criterion.to(torch.device("cuda:0"))
        #color_crit = color_crit.to(torch.device("cuda:0"))

    # load checkpoint/load model
    star_n_iter = 0
    start_epoch = 0
    if hparams["resume"]:
        checkpoint = torch.load(hparams["resume_train"]) ## look at what to load
        #start_epoch = checkpoint['epoch']
        #start_n_iter = checkpoint['n_iter']
        Net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print("last checkpoint restored")


    param_size = 0
    for param in Net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in Net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    
    
    print('model size: {:.3f}MB'.format(size_all_mb))

    training = Train(hparams, Net, optimizer, scheduler,criterion)

    #training.train(dataloader, datasetname, max epochs, start epochs)
    training.train(dataloader_cropnh_haze, "cropnh_haze",hparams["epochs_cropnh_haze"])
    #training.train(dataloader_cropo_haze, "cropo_haze",hparams["epochs_cropo_haze"],600)
    training.train(dataloader_cityscapes, "cityscapes", hparams["epochs_cityscapes"],600)
    training.train(dataloader_o_haze, "o_haze",hparams["epochs_o_haze"],800)
    training.train(dataloader_cityscapes, "cityscapes2", hparams["epochs_cityscapes2"],1000)
    training.train(dataloader_nh_haze, "nh_haze",hparams["epochs_nh_haze"],1200)


    myutils.print_network(Net, hparams)
