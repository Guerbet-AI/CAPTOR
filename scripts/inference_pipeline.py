import totalsegmentator_inference
import preprocessing 
import inference
import os
import warnings
from captor.utils.utils_dict import regions
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the whole pipeline : total segmentator inference, preprocessing, training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:

        # Extract with custom paths
        python whole_pipeline.py 
            --XXX XXX \\
          
            """
        )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./images",
        help="Path where the nifti images are saved"
    )
    parser.add_argument(
        "--segmentations_dir",
        type=str,
        default="./TS_outputs",
        help="Path to save the totalsegmentator outputs"
    )
    parser.add_argument(
        "--do_segmentations",
        type=str,
        default='True',
        help="Whether to do the segmentations if not already done"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of workers"
    )

    parser.add_argument(
        "--path_dict_std_to_region_to_series_reports",
        type=str,
        default="./dict_std_to_region_to_series_reports.pkl",
        help="""Path to the dict containing { study : 'series' : {'region' : series_name },
                                                      'report' : {'region' : report_part },
                                                      'set' : TRAIN/VAL/TEST,
                                                      'ct_types' : {series_1 : ct_type, series_2:ct_type ... }}
            ct type in ['THORAX', 'ABDOMEN', 'PELVIS', 'TA', 'AP', 'TAP', 'UNKNOWN]  
        """
    )

    parser.add_argument(
        "--path_global_dict_crop_centers",
        type=str,
        default="./global_dict_crop_centers.pkl",
        help="Path to the crop centers coordinates of each image"
    )
    parser.add_argument(
        "--do_preprocessing",
        type=str,
        default='True',
        help="Whether to do the preprocessing if not already done"
    )
    parser.add_argument(
        "--dir_save_inference",
        type=str,
        default="./",
        help="Dir to save inferences"
    )
    
    parser.add_argument(
        "--path_clinical_long_former",
        type=str,
        default=None,
        help="Path of initial model weights for text encoder"
    )

    parser.add_argument(
        "--path_weights_resnet152",
        type=str,
        default=None,
        help="Path of initial model weights for image encoder"
    )

    parser.add_argument(
        "--weights_dir",
        type=str,
        default='Guerbet-AI/CAPTOR',
        help="path of initial weights directory if downloaded locally"
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default='CAPTOR_30K',
        help="CAPTOR_30K or CAPTOR_100K"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=18,
        help="Batch size"
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory"
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.segmentations_dir):
        os.makedirs(args.segmentations_dir)


    if args.do_segmentations=='True':
        print('\n', "="*50, 'TotalSegmentator inference', "="*50)
        totalsegmentator_inference.main(images_dir=args.images_dir, output_dir=args.segmentations_dir, path_dict_std_to_region_to_series_reports=args.path_dict_std_to_region_to_series_reports)
    else:
        print('\n', "="*50, 'TotalSegmentator inference skipped', "="*50)

    if args.do_preprocessing=='True':
        print('\n', "="*50, 'Preprocessing', "="*50)
        preprocessing.main(images_dir=args.images_dir, segmentations_dir=args.segmentations_dir, workers=args.workers, path_dict_std_to_region_to_series_reports=args.path_dict_std_to_region_to_series_reports, output_path=args.path_global_dict_crop_centers)
    else:
        print('\n', "="*50, 'Preprocessing skipped', "="*50)


    print('\n', "="*50, 'Inference', "="*50)
    inference.main(images_dir=args.images_dir,
             path_global_dict_crop_centers=args.path_global_dict_crop_centers, 
             path_dict_std_to_region_to_series_reports=args.path_dict_std_to_region_to_series_reports, 
             path_clinical_long_former=args.path_clinical_long_former,
             path_weights_resnet152=args.path_weights_resnet152,
             weights_dir=args.weights_dir,
             model_type=args.model_type,
             dir_save_inference=args.dir_save_inference,
             cache_dir=args.cache_dir,
             batch_size=args.batch_size,
             regions=regions)
