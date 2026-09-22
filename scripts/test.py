import pickle
import os 
import json
from captor.utils.utils_dict import regions
import numpy as np

path_save_dict = '/Users/selenajm/Documents/Data/test_code/dict_std_to_region_to_series_reports.pkl'
path_images = '/Users/selenajm/Documents/Data/test_code/data/'

path_dict_pat_volumes = '/Users/selenajm/Documents/Data/report_filtering/data/dict_pat_volumes_wholeTAP.json'
path_dict_pat_volumes_wholeTAP_regions_to_vol= '/Users/selenajm/Documents/Data/report_filtering/data/dict_pat_volumes_wholeTAP_regions_to_vol.json'
dict_pat_volumes = json.load(open(path_dict_pat_volumes, 'rb'))
dict_pat_volumes_regions_to_vol = json.load(open(path_dict_pat_volumes_wholeTAP_regions_to_vol, 'rb'))
outfile_path_pkl = '/Users/selenajm/Documents/Data/report_filtering/data/mistral_splitting_new.pkl'
mistral_output = pickle.load(open(outfile_path_pkl, "rb"))
print(list(dict_pat_volumes.keys())[:20])

dict_std_to_region_to_series_reports = {}
stds_downloaded = np.unique([vol[:vol.index('_series')] for vol in os.listdir(path_images) if 'series' in vol]).tolist()
print(stds_downloaded)
for i_std, std in enumerate(stds_downloaded):
    dict_std_to_region_to_series_reports[std] = {'series':{}, 'report':{}, 'ct_types':{}}
    # Reports 
    res = json.loads(mistral_output[std])
    for reg in regions:
        dict_std_to_region_to_series_reports[std]['report'][reg] = res[reg]['region_report']
        dict_std_to_region_to_series_reports[std]['series'][reg] = dict_pat_volumes_regions_to_vol[std][reg]
    for series in dict_std_to_region_to_series_reports[std]['series'].values():
        dict_std_to_region_to_series_reports[std]['ct_types'][series] = dict_pat_volumes[std][series]
    
    # if i_std <= len(os.listdir(path_images))*0.7:
    #     dict_std_to_region_to_series_reports[std]['set'] = 'TRAIN'
    # elif i_std <= len(os.listdir(path_images))*0.8:
    #     dict_std_to_region_to_series_reports[std]['set'] = 'VAL'
    # else:
    #     dict_std_to_region_to_series_reports[std]['set'] = 'TEST'

    if i_std <= len(stds_downloaded)*0.7:
        dict_std_to_region_to_series_reports[std]['set'] = 'TRAIN'
    else:
        dict_std_to_region_to_series_reports[std]['set'] = 'VAL'

with open(path_save_dict, 'wb') as f :
    pickle.dump(dict_std_to_region_to_series_reports, f)


# uv run python scripts/train_pipeline.py --images_dir /Users/selenajm/Documents/Data/test_code/data --segmentations_dir /Users/selenajm/Documents/Data/test_code/segmentations --path_dict_std_to_region_to_series_reports /Users/selenajm/Documents/Data/test_code/dict_std_to_region_to_series_reports.pkl --path_global_dict_crop_centers /Users/selenajm/Documents/Data/test_code/global_dict_crop_centers.pkl --dir_save_model /Users/selenajm/Documents/Data/test_code --exp_name test --path_clinical_long_former /Users/selenajm/Documents/duonco-tap-foundation-train/image_processing/Merlin/clinical_longformer_local --path_weights_resnet152 /Users/selenajm/Documents/duonco-tap-foundation-train/image_processing/Merlin/weights_resnet152.pth --cache_dir /Users/selenajm/Documents/Data/test_code/cache --epochs 2 --do_segmentations False
# uv run python scripts/inference_pipeline.py --images_dir /Users/selenajm/Documents/Data/test_code/data --segmentations_dir /Users/selenajm/Documents/Data/test_code/segmentations --path_dict_std_to_region_to_series_reports /Users/selenajm/Documents/Data/test_code/dict_std_to_region_to_series_reports.pkl --path_global_dict_crop_centers /Users/selenajm/Documents/Data/test_code/global_dict_crop_centers.pkl --path_clinical_long_former /Users/selenajm/Documents/duonco-tap-foundation-train/image_processing/Merlin/clinical_longformer_local --path_weights_resnet152 /Users/selenajm/Documents/duonco-tap-foundation-train/image_processing/Merlin/weights_resnet152.pth --path_initial_model_weights /Users/selenajm/Documents/Data/test_code/test/test_best.ckpt --cache_dir /Users/selenajm/Documents/Data/test_code/cache --do_segmentations False --do_preprocessing False --dir_save_inference /Users/selenajm/Documents/Data/test_code/inferences