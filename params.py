hparams = {
    "seed": 123,
    "gpus": 1,
    "gpu_mode": False,  #True
    "crop_size": None,
    "resume": False,
    "test_data_path": "D:/Data/Data/dehaze/SOTS/outdoor/",
    "train_data_path": "D:/Data/Data/dehaze/prepared/", #C:/Data/dehaze/prepared/
    "augment_data": True,
    "epochs_o_haze": 1000,
    "epochs_nh_haze": 2000,
    #"epochs_cropo_haze": 300,
    "epochs_cropnh_haze": 600,
    "epochs_cityscapes": 800,
    "epochs_cityscapes": 1200,
    "batch_size": 4,
    "gen_lambda": 1,
    "color_lambda": 0.5,
    "pseudo_lambda": 0.5,
    "down_deep": False,
    "threads": 0,
    "height":640, #1280, 512, 288 niedrigste zahl = 248
    "width":360,    #720, 288, 288 solange durch 8 teilbar & >= 248
    "lr":1e-04,
    "beta1": 0.9595,
    "beta2": 0.9901,
    "mhac_filter": 64,  
    "mha_filter": 32,    
    "num_mhablock": 6,  
    "num_mhac":9, 
    "num_parallel_conv": 2,
    "kernel_list": [3,5,7],
    "pad_list": [4,12,24],
    "start_epoch": 509,
    "save_folder": "./weights/",
    "model_type": "Dehaze",
    "scale_factor": 1,
    "snapshots": 25,
    "pseudo_alpha": 1,
    #"pseudo_alpha": 0.6,
    #"hazy_alpha": 0.7,
    "hazy_alpha": 1,
    "resume_train": "./weights/649nh_haze_Dehaze.pth"
}

#sec best params 93.97999572753906.
# mhac_filter: 64
# mha_filter: 32
# num_mhablock: 6
# num_mhac: 7
# gen_lambda: 0.5
# pseudo_lambda: 0.7000000000000001
# pseudo_alpha: 1.0
# hazy_alpha: 1.0