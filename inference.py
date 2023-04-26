import torch
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor
from models import *
import torchvision.transforms as T
from torchvision import transforms, utils
from PIL import Image
import argparse
import time
import os
from params import hparams

parser = argparse.ArgumentParser(description='PyTorch Dehaze')
parser.add_argument('--modelpath', type=str, default="weights/999Dehaze.pth", help=("path to the model .pth files"))
parser.add_argument('--inferencepath', type=str, default='D:/Data/Smoke/test/hazy/', help=("Path to image folder"))
parser.add_argument('--gtpath', type=str, default=None, help=("Path to image folder"))
parser.add_argument('--gpu_mode', type=bool, default=False, help=('enable cuda'))
    
if __name__ == '__main__':
    psnrrating = []

    opt = parser.parse_args()

    i = 0
    PATH = opt.modelpath
    #imagespath = (opt.inferencepath + opt.imagename)
    imagespath = os.listdir(opt.inferencepath)
    gtpath = os.listdir(opt.gtpath)
    if not os.path.isdir("results"):
        os.mkdir("results")

    for i,imagepath in enumerate(imagespath):  
        if gtpath is not None:
            gtimg = Image.open(opt.gtpath + gtpath[i])
            gtimg = gtimg.resize((int(hparams["height"]), int(hparams["width"])))
            gtimg.save('results/'+str(i)+"_GT"+'.png')
        image = Image.open(opt.inferencepath + imagepath)
        image = image.resize((int(hparams["height"]), int(hparams["width"])))
        image.save('results/'+str(i)+'.png')


        transformtotensor = transforms.Compose([transforms.ToTensor()])
        imagetens = transformtotensor(image)
        imagetens = imagetens.unsqueeze(0)
        imagetens= imagetens.to(torch.float32)

        model=Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"],hparams["down_deep"] ,hparams["gpu_mode"], hparams["scale_factor"], hparams["pseudo_alpha"], hparams["hazy_alpha"], size=(hparams["height"], hparams["width"]))

        if opt.gpu_mode == False:
            device = torch.device('cpu')

        if opt.gpu_mode:
                device = torch.device('cuda')
        
        checkpoint = torch.load(PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        start = time.time()
        transform = T.ToPILImage()
        imagetens = imagetens.to(device)
        times = []
        allproctime = 0
        start = time.time()
        out, pseudo = model(imagetens)

        end = time.time()
        proctime = round(end -start, 4)

        out = transform(out.squeeze(0))
        if opt.gtpath is not None:
            psnrate = myutils.psnr(image,out)
            psnrrating.append(psnrate)

        print("Inferencetime is: " + str(proctime) + "seconds")
        out.save('results/'+str(i)+'_Dehaze.png')
        pseudo = transform(pseudo.squeeze(0))
        pseudo.save('results/'+str(i)+'_Pseudo.png')
    
    if opt.gtpath is not None:
        print("------------")
        print("PSNR Rating:")
        psnrrating.sort()
        print("PSNR min: " + str(psnrrating[0]))
        print("PSNR max: " + str(psnrrating[-1]))
