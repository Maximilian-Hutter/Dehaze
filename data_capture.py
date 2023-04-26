# develop programm to generate a real world dataset (not synthesized)

import cv2
from threading import Thread
import time
from params import hparams
from models import Dehaze
import torch
from torchvision import transforms
import numpy
import argparse


class CamStream:
    def __init__(self, stream_id=0):

        self.stream_id = stream_id
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is initialized to False 
        self.stopped = True
        # thread instantiation  
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        self.stopped = False
        self.t.start()

    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            #time.sleep(7)
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method to return latest read frame 
    def read(self):
        return self.frame

    # method to stop reading frames 
    def stop(self):
        self.stopped = True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch Dehaze Datastream')
    parser.add_argument('--modelpath', type=str, default="weights/999Dehaze.pth", help=("path to the model .pth files"))
    opt = parser.parse_args()

    model=Dehaze(hparams["mhac_filter"], hparams["mha_filter"], hparams["num_mhablock"], hparams["num_mhac"], hparams["num_parallel_conv"],hparams["kernel_list"], hparams["pad_list"],hparams["down_deep"] ,hparams["gpu_mode"], hparams["scale_factor"], hparams["pseudo_alpha"], hparams["hazy_alpha"], size=(hparams["height"], hparams["width"]))
    checkpoint = torch.load(opt.modelpath)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    webcam_stream = CamStream(stream_id = 0) # 0 id for main camera
    webcam_stream.start()

    while True :

        if webcam_stream.stopped is True:
            break
        else :
            frame = webcam_stream.read()

        frame = cv2.resize(frame,(640,360),interpolation = cv2.INTER_AREA)
        transformtotensor = transforms.Compose([transforms.ToTensor()])
        frame = transformtotensor(frame)
        frame = frame.unsqueeze(0)
        frame= frame.to(torch.float32)

        pretime = time.time()
        frame,_ = model(frame)
        postime = time.time()
        print(round(postime - pretime,4))
        transform = transforms.ToPILImage()
        frame = transform(frame.squeeze(0))
        frame = numpy.array(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        #time.sleep(60)


    webcam_stream.stop()
    cv2.destroyAllWindows()