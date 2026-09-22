import torch
import os
import warnings
import numpy as np
import pickle
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import nibabel as nib

from captor.utils.utils_preprocessing import *
from captor.utils.utils_dict import regions, segmentations_to_exclude, segmentations_for_batch_region, dict_idx_segmentations, region_crop_organ_centers, splits_corresponding_to_body_part



warnings.filterwarnings("ignore")
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
torch.manual_seed(0)

#=================================================== Preprocessing function ==========================================
def process_one_image(image_path, ct_type_dict, segmentations_dir, transform):
    if os.path.basename(image_path).replace('.nii.gz', '') in ct_type_dict.keys():
        try:
            ct_type = ct_type_dict[os.path.basename(image_path).replace('.nii.gz', '')]
            dict_crop_centers = {}
            regions_done = []

            transformed = transform({'image' : image_path})
            image_meta_dict = transformed["image_meta_dict"]
            
            key = os.path.basename(image_path)
            segmentation_path = f'{segmentations_dir}/{key.replace('.nii.gz', '.nii')}'
            seg_img = nib.load(segmentation_path)
            
            resampled_seg, is_LPS = align_seg(seg_img, image_meta_dict, segmentation_path)

            for reg in splits_corresponding_to_body_part[ct_type]:
                if reg not in regions_done : #musculoskeletal and vasculature are possible with each body parts, take only the first one, like whole_report
                    segmentation = get_input_tensor_before_crop(reg, ct_type, resampled_seg, image_meta_dict, segmentations_for_batch_region, segmentations_to_exclude, dict_idx_segmentations, is_LPS)

                    if segmentation is not None :
                        if (reg != 'whole_report'): #(reg not in splits_to_exclude[ct_type]) and 
                            crop_center = get_joint_center(segmentation=segmentation, labels=region_crop_organ_centers.get(reg))
                        else:
                            crop_center = np.array([-1, -1, -1])
                    else : 
                        crop_center = np.array([-1, -1, -1])

                    dict_crop_centers[reg] = crop_center
                        
                    regions_done += [reg]
            try:
                # all the regions invisible in all the images for the std, and the region 'whole_report'
                reg = 'whole_report'
                segmentation = get_input_tensor_before_crop(reg, ct_type, resampled_seg, image_meta_dict, segmentations_for_batch_region, segmentations_to_exclude, dict_idx_segmentations, is_LPS)
                if segmentation is not None :
                    if (reg != 'whole_report'): #(reg not in splits_to_exclude[ct_type]) and 
                        crop_center = get_joint_center(segmentation=segmentation, labels=region_crop_organ_centers.get(reg))
                    else:
                        crop_center = np.array([-1, -1, -1])
                else : 
                    crop_center = np.array([-1, -1, -1])            # print(f'Region {reg} done in {time.time() - reg_time}')
            except Exception as e: 
                print(e)
            
            regions_not_in_any_image = list(set(regions) - set(regions_done)) + ['whole_report']
            for reg in regions_not_in_any_image:
                dict_crop_centers[reg] = crop_center

        except Exception as e:
            print(os.path.basename(image_path), e)
    else : 
        dict_crop_centers = None
    return image_path, dict_crop_centers


def main(images_dir,segmentations_dir, workers, path_dict_std_to_region_to_series_reports, output_path):
    if os.path.isfile(output_path):
        global_dict_crop_centers = pickle.load(open(output_path, "rb"))
    else:
        global_dict_crop_centers = {}
    
    dict_std_to_region_to_series_reports = pickle.load(open(path_dict_std_to_region_to_series_reports, "rb"))
    
    ct_type_dict = {}
    for study_data in dict_std_to_region_to_series_reports.values():
        ct_type_dict.update(study_data["ct_types"])

    image_list = get_list_final(images_dir, segmentations_dir, global_dict_crop_centers, ct_type_dict)
    # Function for CT-RATE because of the data architecture
    # list_final = get_list_final_CTRATE(images_dir,  segmentations_dir, global_dict_crop_centers) 

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    transform = ImageTransforms_with_segmentations

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_image, img, ct_type_dict, segmentations_dir, transform) for img in image_list]
        for i, fut in enumerate(
            tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Crop centers",
                unit="image"
            ),
            start=1
        ):
            try:
                image_path, dict_crop_centers = fut.result()
                if dict_crop_centers is not None:
                    key = os.path.basename(image_path)
                    if '.nii' in key:
                        key = key[:key.index('.nii')]
                        global_dict_crop_centers[key] = dict_crop_centers

                # print(f"Done: {os.path.basename(image_path)}")
                
                if i%10 == 0 :
                    with open(output_path, "wb") as f:
                        pickle.dump(global_dict_crop_centers, f)
                    
            except Exception as e:
                print(e)
                continue

        with open(output_path, "wb") as f:
            pickle.dump(global_dict_crop_centers, f)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute crop centers for each region for each study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:

        # Extract with custom paths
        python preprocessing.py 
            --ct_type_dict_path ./ct_type_dict.pkl \\
            --segmentations_dir ./TS_outputs \\
            --images_dir ./images \\
            --workers 10 \\
            --output_path ./dict_crop_centers.pkl
            
            """
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
        "--segmentations_dir",
        type=str,
        default="./TS_outputs",
        help="Path to totalsegmentator inferences"
    )

    parser.add_argument(
        "--images_dir",
        type=str,
        default="./images",
        help="Path where the nifti images are saved"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of workers"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./dict_crop_centers.pkl",
        help="Path to save the output"
    )
    args = parser.parse_args()



    # ================================================RUN the preprocessing ===============================================================
    main(args.images_dir, args.segmentations_dir, args.workers, args.path_dict_std_to_region_to_series_reports, args.output_path)
