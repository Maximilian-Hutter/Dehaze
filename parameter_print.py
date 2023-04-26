from models import *
from basic_models import *
import numpy as np
import torch
import torch.nn as nn
import socket
import torch.backends.cudnn as cudnn
from torchsummary import summary
import time
from params import hparams

if __name__ == "__main__":

    np.random.seed(hparams["seed"])    # set seed to default 123 or opt
    torch.manual_seed(hparams["seed"])
    torch.cuda.manual_seed(hparams["seed"])
    gpus_list = range(hparams["gpus"])
    hostname = str(socket.gethostname)
    cudnn.benchmark = True

    # defining shapes


    Net = Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"],hparams["down_deep"] ,hparams["gpu_mode"], hparams["scale_factor"], hparams["pseudo_alpha"], hparams["hazy_alpha"], size=(hparams["height"], hparams["width"]))
    

    start = time.time()
    summary(Net, (3, 640, 360), device="cpu")
    end = time.time()

    proctime = end-start
    print(proctime)

    # pytorch_params = sum(p.numel() for p in Net.parameters())
    # print("Network parameters: {}".format(pytorch_params))

    # def print_network(net):
    #     num_params = 0
    #     for param in net.parameters():
    #         num_params += param.numel()
    #     print(net)
    #     print('Total number of parameters: %d' % num_params)

    # print('===> Building Model ')
    # Net = Net


    # print('----------------Network architecture----------------')
    # print_network(Net)
    # print('----------------------------------------------------')