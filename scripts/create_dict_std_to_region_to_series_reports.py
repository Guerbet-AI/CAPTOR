import os 
import random
from captor.utils.utils_dict import regions
import pickle

'''
This script gives a minimal code to generate the needed pickle file for inference and training pipeline. 

Modify this code to fit your data, this minimal code only fits for monoregion data (for example TAP only data) 
without any studies with several series to exploit (studies with a chest CT and an abdominal-pelvis CT for example)
'''
### Parameters 
images_dir = './images'
reports_dir = './reports'
image_types = 'TAP'
path_save_dict = './dict_std_to_region_to_series_reports'
# region = ['chest', ...] if you want to select only a subset of regions


list_images = os.listdir(images_dir)
random.seed(42)

images_shuffled = random.sample(list_images, len(list_images))

n = len(images_shuffled)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

train_set = images_shuffled[:n_train]
val_set = images_shuffled[n_train:n_train + n_val]
test_set = images_shuffled[n_train + n_val:]


dict_std_to_region_to_series_reports = {}

for i_std, std in enumerate(list_images):
    dict_std_to_region_to_series_reports[std] = {'series':{}, 'report':{}, 'ct_types':{}}
    # Reports, change to whatever you need to load the report of part of report associated to a region
    with open(os.path.join(reports_dir, std), "r") as f:
        report_content = f.read()

    for reg in regions:
        dict_std_to_region_to_series_reports[std]['report'][reg] = report_content
        dict_std_to_region_to_series_reports[std]['series'][reg] = std # change to whatever you need to get the right image for the region
    for series in dict_std_to_region_to_series_reports[std]['series'].values():
        dict_std_to_region_to_series_reports[std]['ct_types'][series] = 'TAP' # change to whatever you need depending on your image types
    

    if std in test_set:
        dict_std_to_region_to_series_reports[std]['set'] = 'TEST'
    elif std in val_set:
        dict_std_to_region_to_series_reports[std]['set'] = 'VAL'
    else:
        dict_std_to_region_to_series_reports[std]['set'] = 'TRAIN'


with open(path_save_dict, 'wb') as f :
    pickle.dump(dict_std_to_region_to_series_reports, f)