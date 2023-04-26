# combine base model: https://arxiv.org/pdf/2111.09733v1.pdf with large kernel structure https://arxiv.org/pdf/2209.01788v1.pdf
# Shallow Layers using the Large Kernel: use LKD before SHA in the MHA
# see what happens when you reaplce the attention in CoT with the CSDLKCB
from numpy import outer
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from basic_models import *
import myutils
# from fightingcv_attention.attention.CoTAttention import *

class CoT(nn.Module):
    def __init__(self, in_feat=512,kernel=3):
        super().__init__()
        self.in_feat=in_feat
        self.kernel_size=kernel

        self.key_embed=nn.Sequential(
            nn.Conv2d(in_feat,in_feat,kernel_size=kernel,padding=kernel//2,groups=4,bias=False),
            nn.InstanceNorm2d(in_feat),
            nn.ELU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(in_feat,in_feat,1,bias=False),
            nn.InstanceNorm2d(in_feat)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*in_feat,2*in_feat//factor,1,bias=False),
            nn.InstanceNorm2d(2*in_feat//factor), # BN to IN
            nn.ELU(), # ReLU to ELU
            nn.Conv2d(2*in_feat//factor,kernel*kernel*in_feat,1)
        )


    def forward(self, x):   # modified from xmu-xiaoma666/External-Attention-pytorch

        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat((k1,x),dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)

        return k1+k2


class TailModule(nn.Module):
    def __init__(self, in_feat, out_feat, kernel, padw, padh):
        super(TailModule, self).__init__()
        pad = (padw,padw,padh,padh)
        self.padding = nn.ReflectionPad2d(pad)
        self.conv1 = DLKCB(in_feat, out_feat, 3,group=1)
        self.elu = nn.ELU()
        self.conv2 = nn.Conv2d(out_feat, out_feat, 3,1,1)
        self.lrelu = nn.LeakyReLU()
    
    def forward(self,x):
        x = self.padding(x)
        x = self.conv1(x)
        x = self.lrelu(x) # if problems persist use a different or no activation

        return x

class MHA(nn.Module):
    def __init__(self, in_feat, out_feat, num_parallel_conv, kernel_list, pad_list, groups, size):
        super(MHA, self).__init__()
        self.num_parallel_conv = num_parallel_conv
        self.kernel_list = kernel_list
        self.pad_list = pad_list
        self.parallel_conv = []
        self.in_feat = in_feat
        
        # for i,_ in enumerate(range(num_parallel_conv), start = 0):
        #     kernel = kernel_list[i]
        #     pad = pad_list[i]
        #     csdlkcb = CSDLKCB(in_feat, out_feat, kernel, pad=pad)
        #     csdlkcb.cuda()
        #     self.parallel_conv.append(csdlkcb)

        self.csdlkcb = CSDLKCB(in_feat, in_feat, kernel = kernel_list[0], pad = pad_list[0],  group=4)
        self.csdlkcb2 = CSDLKCB(in_feat, in_feat, kernel = kernel_list[1], pad = pad_list[1],group=4)
        self.csdlkcb3 = CSDLKCB(in_feat, in_feat, kernel = kernel_list[2], pad = pad_list[2],group=4)

        self.lrelu = nn.LeakyReLU()
        self.convsha = CSDLKCB(in_feat, out_feat,kernel= kernel_list[1], pad=pad_list[1],group=4)
        self.sha = SHA(in_feat, out_feat, groups, size=size)

    def forward(self,x):
        #res = x
        # for i in range(self.num_parallel_conv):

        #     conv = self.parallel_conv[i]
        #     par_out = conv(par_out)
        #     x = torch.add(par_out,x)

        x1 = self.csdlkcb(x)
        x2 = self.csdlkcb2(x)
        x3 = self.csdlkcb3(x)
        x = torch.add(x,x1)
        #x = torch.div(x, x.max())
        x = torch.add(x, x2)
        x = torch.add(x,x3)
        #x = torch.div(x, x.max())

        

        x = self.lrelu(x)
        x = self.convsha(x)
        #x = nn.functional.relu(x)
        x,_ = self.sha(x)
        #x = torch.add(res,x)
        return x

class MHAC(nn.Module):
    def __init__(self, in_feat, inner_feat, out_feat, num_parallel_conv, kernel_list, pad_list, groups, kernel = 3,size=(None,None)):
        super(MHAC, self).__init__()

        self.mha = MHA(in_feat, in_feat, num_parallel_conv, kernel_list, pad_list, groups, size=size)
        self.cot = CoT(in_feat)
        self.aff = AdaptiveFeatureFusion()

    def forward(self,x):

        mhaout = self.mha(x)
        cotout = self.cot(x)
        x = self.aff(mhaout, cotout)
        
    
        return x
        
class SHA(nn.Module):
    def __init__(self, in_feat, out_feat, groups, kernel = 3, downsample = False, size = (None,None)):
        super(SHA,self).__init__()
        self.groups = groups
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.size = size
        self.height = size[0]
        self.width = size[1]

        # might be wrong
        # self.avgh = nn.AvgPool2d((kernel,1),stride=1) # kernel of size 1 horizontaly and 0 verticaly
        # self.maxh = nn.MaxPool2d((kernel,1),stride=1)

        # self.avgv = nn.AvgPool2d((1,kernel),stride=1)
        #self.maxv = nn.MaxPool2d((1,kernel),stride=1)
        #
        self.avgh = nn.AdaptiveAvgPool2d((None,1)) # kernel of size 1 horizontaly and 0 verticaly
        self.maxh = nn.AdaptiveMaxPool2d((None,1))

        self.avgv = nn.AdaptiveAvgPool2d((1,None))
        self.maxv = nn.AdaptiveMaxPool2d((1,None))

        self.shuffle = ChannelShuffle(groups)

        self.relu6 = nn.ReLU6()
        self.downsample = downsample
        if downsample is True:
            stride = 2
        else: 
            stride = 1

        self.conv1 = ConvBlock(in_feat,in_feat, pad=1)
        self.conv2 = ConvBlock(in_feat,out_feat, stride = stride, pad=1)

        self.sigmoid = nn.Sigmoid()

        self.convres = ConvBlock(in_feat, out_feat)
        self.down = nn.Upsample(scale_factor=0.5)
    def forward(self,x):
        res = x
        
        havg = self.avgh(x)
        hmax = self.maxh(x)
        h = torch.add(havg, hmax)
        h = F.relu(h)
        h = torch.div(h,h.max())

   

        vavg = self.avgv(x)
        vmax = self.maxv(x)
        v = torch.add(vavg, vmax)
        v = F.relu(v)
        v = torch.div(v,v.max())
        

        #h = F.pad(h, (0,0,2,0), "constant",0)
        #v = F.pad(v, (0,2), "constant",0)

        batch, channels, height, width = v.size()
        v = v.view(batch, channels, width, height)

        x = torch.cat((h,v), dim= 2)
        x = self.shuffle(x)
        
        x = self.conv1(x)
        x = self.relu6(x)

        x = torch.split(x,int(self.height), dim = 2)

        x1 = x[0]
        x2 = x[1]
        batch, channels, height, width = x1.size()
        x1 = x1.view(batch, channels, width, height)


        x1 = self.conv2(x1)
        x2 = self.conv2(x2)
        
        x = torch.mul(x1,x2)
        shares = x
        x = self.sigmoid(x)

        if self.downsample is True:
            res = self.down(res)

        if self.in_feat is not self.out_feat:
            res = self.convres(res)

        out = torch.add(x,res)

        return out,shares

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self):
        super(AdaptiveFeatureFusion, self).__init__()
        # image was of density estimation

        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()

    def forward(self,x, y):


        # x = x + abs(x.min())
        # x = torch.div(x,x.max())

        # y = y + abs(y.min())
        # y = torch.div(y,y.max())

        # print(x.max())
        # print(x.min())
        # print(y.max())
        # print(y.min())
        # print("--------------")

        x = torch.mul(x,y)
        x = x + abs(x.min())
        x = torch.div(x, x.max())
        #x = torch.mul(self.sig1(x),self.sig2(y))
        #x = torch.div(x,x.max())
         
        
        return x

class DensityEstimation(nn.Module):
    def __init__(self, in_feat, kernel, groups, padw, padh,first_conv_feat=64, size=(640,360)):
        super(DensityEstimation, self).__init__()

        # path # conv -> reflective pad, -> sha -> conv -> sigmoid
        self.csdlkcb = CSDLKCB(in_feat,64,group=3)
        self.elu = nn.ELU()
        self.pad = nn.ReflectionPad2d((padw,padw,padh,padh))
        self.sha = SHA(64, 3, groups=1, size=size)
        self.conv1 = DLKCB(64, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x, y):
        res = y
        x = torch.cat((x,y), dim = 1) # might be wrong dim
        x = self.csdlkcb(x)

        x = self.elu(x)

        x,_ = self.sha(x)
        # x = self.conv1(x)

        # x = self.sigmoid(x)
        #x = torch.add(x,x)
        #x = torch.div(x,x.max())
        ##myutils.save_tensorimg(x,"pre_mul_dense")
        xmin = x + x.min()
        xmin = torch.div(x,x.max())
        res = TF.rgb_to_grayscale(res)
        x = torch.mul(xmin, res)
        ##myutils.save_tensorimg(x,"post_mul_dense")

        return x

class Shallow(nn.Module):
    def __init__(self, in_feat, inner_feat, num_mhac, num_parallel_conv, kernel_list, pad_list, size):
        super(Shallow, self).__init__()
        width = size[1]
        height = size[0]
        small_size = (height/2, width/2)
        smaller_size = (height/4, width/4)

        self.conv1 = ConvBlock(in_feat,in_feat)
        self.sha1 = SHA(in_feat, in_feat, in_feat, downsample=True, size=(size[0],size[1]))
        self.conv2 = ConvBlock(in_feat, inner_feat)
        self.sha2 = SHA(inner_feat, inner_feat, 4, downsample=True, size=small_size)

        self.num_mhac = num_mhac
        self.mhac = MHAC(inner_feat, inner_feat, inner_feat, num_parallel_conv, kernel_list, pad_list, 4, size =smaller_size)


        self.up1 = TransposedUpsample(inner_feat, inner_feat)
        self.sha3 = SHA(inner_feat, inner_feat, 4, size=small_size)
        self.up2 = TransposedUpsample(inner_feat, in_feat)
        self.sha4 = SHA(in_feat, in_feat, in_feat,size=(size[0],size[1]))

        self.tail = TailModule(in_feat,in_feat, 3, 0, 0)

    def forward(self,x):
        res = x

        x = self.conv1(x)

        res1 = x
        x,_ = self.sha1(x)

        
        x = self.conv2(x)
        x = nn.functional.relu(x) # might be a bad solution
        res2 = x

        x,_ = self.sha2(x)

        resmhac = x
        for i in range(self.num_mhac):
            x = self.mhac(x)
            x = torch.add(x, resmhac)
        #print(x.min())
        x = self.up1(x)
        #print(x.min())
        #print(x.shape)
        #print(res2.shape)
        x = torch.add(x, res2)

         
        
        x,_ = self.sha3(x)
        x = self.up2(x)


        #res1 = F.interpolate(res1, scale_factor=1.25)
        x = torch.add(x,res1)
        
         
        x,shares = self.sha4(x)
        #shares = x

        x = self.tail(x)

        #res = F.interpolate(res, scale_factor=1.25)
        x = torch.add(x, res)
        x = x + abs(x.min())
        x = torch.div(x,x.max())
        
        return x, shares

class Deep(nn.Module):
    def __init__(self, in_feat, inner_feat, out_feat, num_mhablock, num_parallel_conv, kernel_list, pad_list,down_deep,size):
        super(Deep,self).__init__()
        self.down_deep = down_deep
        self.num_mhablock = num_mhablock
        if down_deep is True:
            stride = 2
        else:
            stride = 1

        self.aff = AdaptiveFeatureFusion()
        self.conv1 = ConvBlock(in_feat, inner_feat)

        self.mha = MHA(inner_feat, inner_feat, num_parallel_conv, kernel_list, pad_list, 1, size)
        self.norm = nn.InstanceNorm2d(inner_feat)
        self.norm2 = nn.InstanceNorm2d(in_feat)
        self.up = TransposedUpsample(inner_feat, inner_feat)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()        

        self.tail = TailModule(inner_feat, out_feat, 3, 0, 0)

    def forward(self,x, dense):
        x = self.aff(x, dense)
        #myutils.save_tensorimg(x,"after aff")
        x = self.conv1(x)

        resmha = self.relu(x)
        
        for i in range(self.num_mhablock):
            x = self.mha(x)
            x = torch.add(x,resmha)
            x = self.elu(x)

           

        x = self.tail(x)
        x = self.norm(x)
        # x = x + x.min()
        # x = torch.div(x,x.max())
        #myutils.save_tensorimg(x,"after afftail")
        ##myutils.save_tensorimg(x,"after tail")

        return x

class Dehaze(nn.Module):
    def __init__(self, mhac_filter = 256, mha_filter = 16,num_mhablock = 10,num_mhac = 8, num_parallel_conv = 2, kernel_list = [3,5,7], pad_list = [4,12,24], down_deep = False,gpu_mode = True, scale_factor = 1, pseudo_alpha = 1, hazy_alpha = 0.5, size=(360,640)):
        super(Dehaze, self).__init__()
        self.pseudo_alpha = pseudo_alpha
        self.hazy_alpha = hazy_alpha

        #self.preconv = DLKCB(3,4,kernel=11, pad=60)
        #self.pseudconv = DLKCB(4,3, kernel=11, pad=60)
        self.shallow = Shallow(3, mhac_filter, num_mhac, num_parallel_conv, kernel_list, pad_list,size) # filter 256
        self.dense = DensityEstimation(6,3, 4, 0, 0, size=size)
        self.aff = AdaptiveFeatureFusion()
        self.deep = Deep(3, mha_filter, 3, num_mhablock, num_parallel_conv, kernel_list, pad_list, down_deep,size) # filter 16

        self.scale_factor = scale_factor
        if scale_factor != 1:
            self.up = TransposedUpsample(4, 4, 11, scale_factor, False)

        self.norm = nn.InstanceNorm2d(3)
            
    def forward(self, hazy):

        convhazy = hazy
        #convhazy = self.preconv(hazy)
        #convhazy = F.relu(convhazy)
        convpseud, shares = self.shallow(convhazy)
        pseudo = convpseud

        convpseud = F.relu(convpseud)
        #pseudo = self.pseudconv(convpseud)
        #myutils.save_tensorimg(shares,"shares")

        density = self.dense(convpseud, convhazy)
        #myutils.save_tensorimg(density,"pre_share")
        density = torch.mul(density, shares)
        #myutils.save_tensorimg(density,"density")

        x = self.aff(convpseud, convhazy)
        #myutils.save_tensorimg(x,"pre_deep")
        x = self.deep(x, density)

        if self.scale_factor != 1:
            x = self.up(x)
            pseudo = F.interpolate(pseudo, scale_factor=self.scale_factor)
            hazy = F.interpolate(hazy, scale_factor=self.scale_factor)

        #print(x.max())
        #print(x.min())
        x = F.relu(x)
        x = torch.div(x,x.max())

        #myutils.save_tensorimg(x, "pre_pseudo")

        x = torch.add(x, pseudo, alpha=self.pseudo_alpha)
        x = F.relu(pseudo)
         
        #myutils.save_tensorimg(x,"pre_hazy")

        x = torch.add(x, hazy, alpha=self.hazy_alpha)
        x = torch.div(x, x.max())
        #myutils.save_tensorimg(x,"out")
        #myutils.save_tensorimg(pseudo,"pseudo")

        return x, pseudo