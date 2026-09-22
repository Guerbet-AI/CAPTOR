import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import time
import os
import warnings
import numpy as np
import random



warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

# ============================================================== Training ====================================================================
def get_image_embedding_merlin_by_region(model, dataloader, device, path_save_embeddings, regions, emb_2048=True):
    start_time = time.time()
    start_local_time = start_time
    loader_dataset = dataloader.dataset

    for reg in regions :
        print(reg)
        loader_dataset.set_region(reg)
        for i_batch, batch in enumerate(dataloader):
            try :
                with torch.no_grad():
                    if emb_2048 :
                        embedding = model(batch["image"].to(device))[0] #, batch["text"]
                    else :
                        embedding = model.encode_image(batch["image"].to(device))[0] #, batch["text"]
                        if len(embedding.shape) == 1:
                            embedding = embedding.unsqueeze(0)
                        # print(embedding.shape)
                    
                torch.cuda.empty_cache()

                names = batch['study_name']
                for i, name in enumerate(names) :
                    if '.nii.gz' in name :
                        names[i] = name[:name.index('.nii.gz')]  
                                    
                for j, tensor in enumerate(embedding):
                    folder_path = f"{path_save_embeddings}/{names[j]}"
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    filename = f"{path_save_embeddings}/{names[j]}/{reg.replace(" ", "_")}.pt"
                    torch.save(tensor.cpu(), filename)
                        
            except Exception as e:
                print(f"Error: {e}")
                continue

            print(f"Batch : {i_batch}/{len(dataloader)}, batch time : {time.time() - start_local_time}, total_time : {time.time() - start_time}")
            start_local_time = time.time()