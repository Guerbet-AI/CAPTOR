import torch
from torch import nn
import warnings
import numpy as np
import pickle

from captor.models.build import MerlinArchitecture
from captor.data.dataloaders import DataLoader

from captor.utils.utils_train import *
from captor.utils.utils_dict import regions

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu" 
torch.manual_seed(0)

def main(images_dir,
         path_global_dict_crop_centers, 
         path_dict_std_to_region_to_series_reports, 
         train_logit_scale,
         path_clinical_long_former,
         path_weights_resnet152,
         path_initial_model_weights,
         cache_dir,
         batch_size,
         epochs,
         dir_save_model,
         exp_name,
         learning_rate,
         patience,
         regions=None):    

    global_dict_crop_centers = pickle.load(open(path_global_dict_crop_centers, "rb"))
    dict_std_to_region_to_series_reports = pickle.load(open(path_dict_std_to_region_to_series_reports, "rb"))

    model = MerlinArchitecture(ImageEmbedding = False, train_logit_scale=train_logit_scale, path_clinical_long_former=path_clinical_long_former, path_weights_resnet152=path_weights_resnet152)
    model.to(device)

    training_set = [study for study, val in dict_std_to_region_to_series_reports.items() if val['set']=='TRAIN']
    val_set = [study for study, val in dict_std_to_region_to_series_reports.items() if val['set']=='VAL']
    train_datalist = [{
                    "image": dict_std_to_region_to_series_reports[study]['series'],
                    "text" : dict_std_to_region_to_series_reports[study]['report'],
                } for study in training_set]
    
    train_loader = DataLoader(
        datalist = train_datalist,
        images_dir = images_dir,
        cache_dir = cache_dir,
        batchsize = batch_size,
        global_dict_crop_centers = global_dict_crop_centers,
        shuffle = True,
        num_workers = 0,
        force_write_cache = False)
    
    val_datalist = [{
                    "image": dict_std_to_region_to_series_reports[study]['series'],
                    "text" : dict_std_to_region_to_series_reports[study]['report'],
                } for study in val_set]
    
    val_loader = DataLoader(
        datalist = val_datalist,
        images_dir=images_dir,
        cache_dir = cache_dir,
        batchsize = batch_size,
        global_dict_crop_centers = global_dict_crop_centers,
        shuffle = True,
        num_workers = 0,
        force_write_cache = False)

    if regions is None:
        stds = list(dict_std_to_region_to_series_reports.keys())
        regions = list(dict_std_to_region_to_series_reports[stds[0]].keys())
    else:
        regions = regions

    model = train_model(
        model=model, 
        train_loader=train_loader, 
        len_train_dataset=len(train_datalist), 
        batch_size=batch_size, 
        nb_epochs=epochs, 
        dir_save_model=dir_save_model, 
        exp_name=exp_name,
        logit_scale=nn.Parameter(torch.ones([]) * np.log(1 / 0.07)), 
        regions=regions, 
        lr=learning_rate,
        path_load_checkpoint=path_initial_model_weights,
        val_loader=val_loader, 
        patience=patience
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Train model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:

        # Extract with custom paths
        python train.py --images_dir ./images --path_dict_std_reports_trainig ./dict_std_reports_trainig.pkl
            
                """
        )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./images",
        help="Path where the nifti images are saved"
    )
    parser.add_argument(
        "--path_dict_std_to_region_to_series_reports",
        type=str,
        default="./dict_std_to_region_to_series_reports.pkl",
        help="""Path to the dict containing { study : 'series' : {'region' : series_name },
                                                      'report' : {'region' : report_part },
                                                      'set' : TRAIN/VAL/TEST,
                                                      'ct_types' : {series_1 : ct_type, series_2:ct_type ... }}
        """
    )

    parser.add_argument(
        "--path_global_dict_crop_centers",
        type=str,
        default="./global_dict_crop_centers.pkl",
        help="Path to the crop centers coordinates of each image"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=18,
        help="Batch size"
    )
    parser.add_argument(
        "--dir_save_model",
        type=str,
        default="./",
        help="Dir to save model weights"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        help="Name of the experiment"
    )
    
    parser.add_argument(
        "--path_clinical_long_former",
        type=str,
        default="./clinical_longformer",
        help="Path of initial model weights for text encoder"
    )

    parser.add_argument(
        "--path_weights_resnet152",
        type=str,
        default="./weight_resnet_152.ckpt",
        help="Path of initial model weights for image encoder"
    )

    parser.add_argument(
        "--path_initial_model_weights",
        type=str,
        default=None,
        help="Path of initial model weights"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory"
    )
    
    parser.add_argument(
        "--train_logit_scale",
        type=bool,
        default=False,
        help="Whether to train the logit scale as a parameter"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00000125,
        help="Learning rate parameter"
    )

    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        nargs="+",
        help="Regions"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs without increase in performance before stopping"
    )
    args = parser.parse_args()
    

    main(images_dir=args.images_dir,
         path_global_dict_crop_centers=args.path_global_dict_crop_centers, 
         path_dict_std_to_region_to_series_reports=args.path_dict_std_to_region_to_series_reports, 
         train_logit_scale=args.train_logit_scale,
         path_clinical_long_former=args.path_clinical_long_former,
         path_weights_resnet152=args.path_weights_resnet152,
         path_initial_model_weights=args.path_initial_model_weights,
         cache_dir=args.cache_dir,
         batch_size=args.batch_size,
         epochs=args.epochs,
         dir_save_model=args.dir_save_model,
         exp_name=args.exp_name,
         learning_rate=args.learning_rate,
         patience=args.patience,
         regions=regions)
    