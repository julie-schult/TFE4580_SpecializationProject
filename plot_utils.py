import torch
import pandas as pd

BUCKET = "./weights/" 

def save_model(name, model, optimizer, epoch, val):
    PATH = BUCKET + "checkpoint_{}_epoch_{}_acc_{}.pt".format(name,epoch+1,val) 
    torch.save({
            'epoch': epoch+1,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'top1': val,
            }, PATH)

    print("New model, weights saved")