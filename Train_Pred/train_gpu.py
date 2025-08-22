import urllib.request
import tarfile
from pathlib import Path
import os
import pickle
import shutil

args = {
    "batch_size": 5, # make this bigger if you are not running on binder
    "epochs": 200,
    "lr": 0.0001,
    "device": "cuda", # set to "cuda" if GPU is available
    "train_dir": Path("train_npy"),
    "val_dir": Path("val_npy"),
    "save_dir": Path("save_npy")
}

args["save_dir"].mkdir(parents = True, exist_ok=True)

from data import GlacierDataset
from torch.utils.data import DataLoader


paths_train = {
    "x": sorted(list(args["train_dir"].glob("x*"))),
    "y": sorted(list(args["train_dir"].glob("y*")))
}
paths_val = {
    "x": sorted(list(args["val_dir"].glob("x*"))),
    "y": sorted(list(args["val_dir"].glob("y*")))
}


train_ds = GlacierDataset(paths_train["x"], paths_train["y"])
train_loader = DataLoader(train_ds, batch_size=args["batch_size"], shuffle=False)

val_ds = GlacierDataset(paths_val["x"], paths_val["y"])
val_loader = DataLoader(val_ds, batch_size=args["batch_size"], shuffle=False)


import torch.optim
from unet import Unet
from train import train_epoch
from train import validate

model = Unet(9, 3, 4, dropout=0.2).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

Loss=[];
val_loss=[]
for epoch in range(args["epochs"]):
    #enuermate i , i%5==0, save the checkpoint model.
    l=train_epoch(model, train_loader, optimizer, args["device"], epoch, args["save_dir"])
    Loss.append([l[0],l[1]])
    val_loss.append(validate(model,val_loader,args["device"]))
    #if epoch%5==0:
    torch.save(model.state_dict(), f"model_{epoch}.pt")
        

torch.save(model.state_dict(), "model.pt")

with open('loss.pkl', 'wb') as f:
    pickle.dump([Loss,val_loss], f)
