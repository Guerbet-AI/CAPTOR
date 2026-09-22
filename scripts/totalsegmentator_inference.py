import os
import subprocess
import tqdm
import time
import pickle

"""
For documentation on totalsegmentator, see https://github.com/wasserth/totalsegmentator

pip install TotalSegmentator 

If you do not have internet access on the machine you want to run TotalSegmentator on:

Install TotalSegmentator [and set up the license] on a machine with internet.
Run TotalSegmentator for one subject on this machine. This will download the weights and save them to ~/.totalsegmentator.
Copy the folder ~/.totalsegmentator from this machine to the machine without internet.
TotalSegmentator should now work also on the machine without internet."""


def launch_ts_segmentations(input_path,output_path):
    command = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_path,
        "--fast",  # Optionnel : accélère l'inférence
        "--ml"
    ]

    
    subprocess.run(command)

def main(images_dir, output_dir, path_dict_std_to_region_to_series_reports):

    dict_std_to_region_to_series_reports = pickle.load(open(path_dict_std_to_region_to_series_reports, "rb"))
        
    ct_type_dict = {}
    for study_data in dict_std_to_region_to_series_reports.values():
        ct_type_dict.update(study_data["ct_types"])

    list_images = [name for name in os.listdir(images_dir) if name.endswith('.nii.gz') if (name[:name.index('.nii')] in ct_type_dict.keys())]
    outputs = os.listdir(output_dir)
    list_seg_done = [name+'.gz' for name in outputs if name.endswith('.nii')]

    list_images = sorted(list(set(list_images) - set(list_seg_done)))


    for file in tqdm.tqdm(list_images, desc="Segmentation", unit="image"):
        start = time.time()

        input_path = os.path.join(images_dir, file)
        output_path = os.path.join(output_dir, file.replace(".nii.gz", ""))

        launch_ts_segmentations(input_path, output_path)

        end = time.time()

        tqdm.tqdm.write(
            f"Images processed : {file} - Time : {end - start:.1f}s"
        )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract masks from totalsegmentator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example:

        # Extract with custom paths
        python totalsegmentator_inference.py --images_dir ./images --output_dir ./TS_outputs
            
                """
        )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./images",
        help="Path where the nifti images are saved"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./TS_outputs",
        help="Path to save the totalsegmentator outputs"
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

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    main(args)