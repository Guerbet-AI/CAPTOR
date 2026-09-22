import torch
import warnings
import numpy as np
import pickle

from captor.models.build import MerlinArchitecture
from captor.data.dataloaders import DataLoader

from captor.utils.utils_inference import *
from captor.utils.utils_train import load_checkpoint
from captor.utils.utils_dict import regions

warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu" 
torch.manual_seed(0)

def main(images_dir,
         path_global_dict_crop_centers, 
         path_dict_std_to_region_to_series_reports, 
         path_clinical_long_former,
         path_weights_resnet152,
         path_initial_model_weights,
         dir_save_inference,
         batch_size,
         cache_dir,
         regions=None,
         embedding_2048=True):    

    global_dict_crop_centers = pickle.load(open(path_global_dict_crop_centers, "rb"))
    dict_std_to_region_to_series_reports = pickle.load(open(path_dict_std_to_region_to_series_reports, "rb"))

    images = os.listdir(images_dir)
    inferences = os.listdir(dir_save_inference)        
    

    model = MerlinArchitecture(ImageEmbedding = embedding_2048, path_clinical_long_former=path_clinical_long_former, path_weights_resnet152=path_weights_resnet152)
    if path_initial_model_weights is not None :
        if 'ckpt' not in path_initial_model_weights:
            model.load_state_dict(torch.load(path_initial_model_weights))
        else:
            ckpt = torch.load(path_initial_model_weights, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
    model.to(device)

    if regions is None:
        stds = list(dict_std_to_region_to_series_reports.keys())
        regions = list(dict_std_to_region_to_series_reports[stds[0]].keys())
    else:
        regions = regions

    for reg in regions : 
        list_studies = [std for std in dict_std_to_region_to_series_reports.keys() if all([name + '.nii.gz' in images for name in dict_std_to_region_to_series_reports[std]['series'].values()]) and (std not in inferences or (std in inferences and reg.replace(" ", "_")+'.pt' not in os.listdir(os.path.join(dir_save_inference, std))))]
        
        datalist = [{
                        "image": dict_std_to_region_to_series_reports[study]['series'],
                        "text" : dict_std_to_region_to_series_reports[study]['report'],
                        "study_name":study
                    } for study in list_studies]
        
        dataloader = DataLoader(
            datalist = datalist,
            images_dir=images_dir,
            cache_dir = cache_dir,
            batchsize = batch_size,
            global_dict_crop_centers = global_dict_crop_centers,
            shuffle = True,
            num_workers = 0,
            force_write_cache = False)
        
        get_image_embedding_merlin_by_region(model, dataloader, device, dir_save_inference, regions=[reg], emb_2048=True)


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
        "--batch_size",
        type=int,
        default=18,
        help="Batch size"
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
        "--dir_save_inference",
        type=str,
        default="./",
        help="Dir to save inferences"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory"
    )

    parser.add_argument(
        "--embedding_2048",
        type=str,
        default="True",
        help="Whether to have embeddings of size 2048 (second to last layer) or 512 (last layer = contrastive layer)"
    )
    
    args = parser.parse_args()
    

    main(images_dir=args.images_dir,
         path_global_dict_crop_centers=args.path_global_dict_crop_centers, 
         path_dict_std_to_region_to_series_reports=args.path_dict_std_to_region_to_series_reports, 
         path_clinical_long_former=args.path_clinical_long_former,
         path_weights_resnet152=args.path_weights_resnet152,
         path_initial_model_weights=args.path_initial_model_weights,
         dir_save_inference=args.dir_save_inference,
         cache_dir=args.cache_dir,
         batch_size=args.batch_size,
         regions=regions,
         embedding_2048=args.embedding_2048=='True'
         )
    