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

def save_checkpoint(path, model, optimizer, scheduler=None, scaler=None,
                    epoch=0, global_step=0):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler=None, scaler=None, map_location="auto"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    random.setstate(ckpt["python_rng_state"])
    np.random.set_state(ckpt["numpy_rng_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])  

    if torch.cuda.is_available() and ckpt["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])  

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)
    return start_epoch, global_step


def eval_model(model, test_loader, logit_scale, regions,
               split_report=True, k=5, other_metrics=False):

    device = next(model.parameters()).device
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    model.eval()
    torch.cuda.empty_cache()

    start_time = time.time()
    start_local_time = start_time
    i_region = 0
    eval_loss = 0.0
    
    if other_metrics:
        pos_sims = []
        neg_sims = []
        # On accumule seulement les scores, pas les features !
        total_hits = 0
        total_samples = 0

    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
        for i_batch, batch in enumerate(test_loader):
            # Select region

            if i_region >= len(regions):
                i_region = 0
            region = regions[i_region]
            i_region += 1

            test_loader.dataset.set_region(region)
            text = batch["text"][region]
            input_tensor = batch['image'].to(device, non_blocking=True)
            outputs = model(input_tensor, text)
            batch_size_cur = input_tensor.size(0)

            image_features = outputs[0].detach()
            text_features = outputs[2].detach()
            del outputs

            scale = logit_scale.detach().exp()
            sims = image_features @ text_features.T
            logits_per_image = scale * sims
            logits_per_text = logits_per_image.t()

            ground_truth = torch.arange(batch_size_cur, dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) +
                          loss_txt(logits_per_text, ground_truth)) / 2.0
            
            eval_loss += total_loss.item()

            if other_metrics:
                # Calcul des métriques sur le batch uniquement
                pos_sims.append(sims.diag().mean().item())
                if sims.size(0) > 1:
                    neg_sims.append(
                        (sims.sum() - sims.diag().sum()).item() /
                        (sims.numel() - sims.size(0))
                    )
                else:
                    neg_sims.append(0.0)
                
                # R@k sur le batch (approximation)
                # ranks = sims.argsort(dim=1, descending=True)
                # targets = torch.arange(batch_size_cur, device=device)
                # hits = (ranks[:, :k] == targets[:, None]).any(dim=1)
                k_eff = min(k, sims.size(1))
                topk = sims.topk(k_eff, dim=1).indices
                targets = torch.arange(batch_size_cur, device=device)
                hits = (topk == targets[:, None]).any(dim=1)
                total_hits += hits.sum().item()
                total_samples += batch_size_cur
                del topk, targets, hits

            # Nettoyage explicite
            del image_features, text_features, sims, logits_per_image, logits_per_text
            del ground_truth, total_loss

            
            print(f"Eval Batch : {i_batch}/{len(test_loader)}, "
                  f"batch time : {time.time() - start_local_time:.4f}")
            
            start_local_time = time.time()

    torch.cuda.empty_cache()
    
    if other_metrics:
        return eval_loss, {
            "cosine_pos": sum(pos_sims) / len(pos_sims),
            "cosine_neg": sum(neg_sims) / len(neg_sims),
            "r@5": total_hits / total_samples 
        }
    else:
        return eval_loss
    

def train_model(model, train_loader, len_train_dataset, batch_size, nb_epochs, dir_save_model, exp_name, logit_scale, regions, lr = 5e-5, val_loader=None, best_val_loss=1e6, patience=10):
    device = next(model.parameters()).device

    path_save_checkpoint = os.path.join(dir_save_model, exp_name)
    if not os.path.exists(path_save_checkpoint):
        os.makedirs(path_save_checkpoint)
    
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr ,betas=(0.9,0.99)) 
            
    start_epoch = 0

    # torch.cuda.empty_cache()
    loader_dataset = train_loader.dataset

    epoch_losses = []
    eval_losses = []
    cos_pos_list = []
    cos_neg_list = []
    r_list=[]
    diff_cos = []

    if val_loader is not None :
        eval_loss, stats = eval_model(model, val_loader, logit_scale, regions, other_metrics=True)
        eval_losses.append(eval_loss)
        if eval_loss < best_val_loss:
            best_val_loss = float(eval_loss)
        #     # torch.save(model.state_dict(), path_save_model[:path_save_model.index('.pt')]+'_best.pt')

        #     save_checkpoint(os.path.join(path_save_checkpoint, f'{exp_name}_best.ckpt'), model, optimizer, scheduler=None, scaler=None, epoch=0)

        print(f"Eval losses : {eval_losses}")
        
        cos_pos_list.append(stats['cosine_pos'])
        cos_neg_list.append(stats['cosine_neg'])
        diff_cos.append(stats['cosine_pos']-stats['cosine_neg'])
        r_list.append(stats['r@5'])
        print(
            f"[Metrics] cosine_pos={stats['cosine_pos']:.3f} | "
            f"cosine_neg={stats['cosine_neg']:.3f} | "
            f"R@5={stats['r@5']:.3f}"
        )
        print(f"Cosine pos list: {cos_pos_list}")
        print(f"Cosine neg list: {cos_neg_list}")
        print(f"Cosine diff list: {diff_cos}")
        print(f"R@5 list: {r_list}")

        model.train()


    nb_without_best = 0
    for epoch in range(start_epoch, nb_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        start_local_time = start_time
        i_region = 0
        for i_batch, batch in enumerate(train_loader):
            if i_region >= len(regions):
                i_region = 0
            region = regions[i_region]
            i_region += 1            
            
            optimizer.zero_grad()
                
            loader_dataset.set_region(region)
            text = batch["text"][region]
            input_tensor = batch['image'].to(device)
            outputs = model(
                input_tensor, 
                text
            )
            batch_size_cur=input_tensor.size(0)

            image_features, text_features = outputs[0], outputs[2]
            
            scale = logit_scale.exp()

            # cosine similarity as logits
            # scalar product between each embedding of image and text --> Cosine simimarity bc the embeddings are normalized
            # use a temperature to control de distribution concentrations 
            logits_per_image = scale * image_features @ text_features.t() # images on the rows and text on columns (or the other way around)
            logits_per_text = logits_per_image.t() # same matrix as logits_per_image but transposed

            # only the text and its image are considered a positive pair
            ground_truth = torch.arange(batch_size_cur,dtype=torch.long,device=device) # vector [0, 1, .., len(batch)-1] : represents the diagonal --> the positive pairs are the same index

            # cross entropy uses a softmax that transforms the logits in probabilities to be the right pairing
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

            
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            
            print(f"Batch : {i_batch+1}/{len(train_loader)}, 'region : '{regions[i_region]}, batch time : {time.time() - start_local_time:.4f}, epoch time : {time.time() - start_time:.4f}, mean loss : {total_loss/batch_size:.4f}")
            start_local_time = time.time()
            # torch.cuda.empty_cache()
        
        if epoch % 5 == 0 : 
            # torch.save(model.state_dict(), path_save_model[:path_save_model.index('.pt')]+f'_{epoch}.pt')

            save_checkpoint(os.path.join(path_save_checkpoint, f'{exp_name}_epoch_{epoch}.ckpt'), model, optimizer, scheduler=None, scaler=None, epoch=epoch)

        print(f"Epoch {epoch+1}/{nb_epochs} - Epoch time {time.time() - start_time:.4f} - Mean loss: {epoch_loss/len_train_dataset:.4f}")
        epoch_losses += [epoch_loss/len_train_dataset]

        print(f'Epoch losses : {epoch_losses}')
        
        if val_loader is not None :
            eval_loss, stats = eval_model(model, val_loader, logit_scale, regions, other_metrics=True)
            eval_losses.append(eval_loss)
            if eval_loss < best_val_loss:
                best_val_loss = float(eval_loss)
                # torch.save(model.state_dict(), path_save_model[:path_save_model.index('.pt')]+'_best.pt')

                save_checkpoint(os.path.join(path_save_checkpoint, f'{exp_name}_best.ckpt'), model, optimizer, scheduler=None, scaler=None, epoch=epoch)

                nb_without_best = 0
            else :
                nb_without_best +=1 
            print(f"Eval losses : {eval_losses}")
            
            cos_pos_list.append(stats['cosine_pos'])
            cos_neg_list.append(stats['cosine_neg'])
            diff_cos.append(stats['cosine_pos']-stats['cosine_neg'])
            r_list.append(stats['r@5'])
            print(
                f"[Metrics] cosine_pos={stats['cosine_pos']:.3f} | "
                f"cosine_neg={stats['cosine_neg']:.3f} | "
                f"R@5={stats['r@5']:.3f}"
            )
            print(f"Cosine pos list: {cos_pos_list}")
            print(f"Cosine neg list: {cos_neg_list}")
            print(f"Cosine diff list: {diff_cos}")
            print(f"R@5 list: {r_list}")

            if nb_without_best > patience:
                print(f'No more patience, left at epoch : {epoch}')
                # torch.save(model.state_dict(), path_save_model)
                save_checkpoint(os.path.join(path_save_checkpoint, f'{exp_name}_epoch_{epoch}.ckpt'), model, optimizer, scheduler=None, scaler=None, epoch=epoch)    
                return model
            
            model.train()
    save_checkpoint(os.path.join(path_save_checkpoint, f'{exp_name}_epoch_{epoch}.ckpt'), model, optimizer, scheduler=None, scaler=None, epoch=epoch)    
    return model
