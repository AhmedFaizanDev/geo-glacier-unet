import urllib.request
import tarfile
from pathlib import Path
import os
import pickle
import shutil

args = {
    "batch_size": 1, # make this bigger if you are not running on binder
    "epochs": 50,
    "lr": 0.0001,
    "device": "cuda", # set to "cuda" if GPU is available
    "train_dir": Path("train_npy"),
    "val_dir": Path("val_npy"),
    "save_dir": Path("save_npy"),
    "model_dir": Path("model")
}

args["save_dir"].mkdir(parents = True, exist_ok=True)

from data import GlacierDataset
from torch.utils.data import DataLoader

paths_val = {
    "x": sorted(list(args["val_dir"].glob("x*"))),
    "y": sorted(list(args["val_dir"].glob("y*")))
}


val_ds = GlacierDataset(paths_val["x"], paths_val["y"])
val_loader = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False)


import torch.optim
from unet import Unet
from train import train_epoch
from train import validate

paths_model = sorted(list(args["model_dir"].glob("model_*pt")))
paths_model_new = [str(x) for x in paths_model]
paths_model=sorted(paths_model_new, key=lambda i: int(os.path.splitext(os.path.basename(i))[0][6:]))

val_loss=[]
for i in paths_model:
  model = Unet(9, 3, 4, dropout=0.2).to(args["device"])
  model.load_state_dict(torch.load(i))
  val_loss.append(validate(model,val_loader,args["device"]))
       
with open('val_loss.pkl', 'wb') as f:
    pickle.dump([val_loss], f)
