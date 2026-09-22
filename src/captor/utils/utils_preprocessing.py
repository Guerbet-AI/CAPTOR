import torch
import os
import numpy as np
import pickle
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor,  wait, FIRST_EXCEPTION
from concurrent.futures.process import BrokenProcessPool
from collections.abc import Sequence

import nibabel as nib

from monai.data.meta_tensor import MetaTensor
from monai.transforms.utils import convert_to_tensor, ensure_tuple
from monai.utils.enums import TraceKeys




from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
    ToTensor,
    BorderPad,
    InvertibleTransform, 
    LazyTransform,
    ResampleToMatchd
)

from captor.utils.utils_dict import regions, segmentations_to_exclude, segmentations_for_batch_region, dict_idx_segmentations, region_crop_organ_centers,splits_corresponding_to_body_part
#=====================================================================================================================================================================

def get_list_final(path_images, segmentations_dir, global_dict_crop_centers,ct_type_dict, nb_images=None):
    list_images = [name[:name.index('.nii')] for name in os.listdir(segmentations_dir) if (name.endswith('.nii')) and (name[:name.index('.nii')] not in list(global_dict_crop_centers.keys())) and (name+'.gz' in os.listdir(path_images)) and (name[:name.index('.nii')] in ct_type_dict.keys())]

    if nb_images is not None :
        list_final = [os.path.join(path_images,name)+'.nii.gz' for name in list_images[:nb_images]]
    else:
        list_final = [os.path.join(path_images,name)+'.nii.gz' for name in list_images]
    return list_final

def get_list_final_CTRATE(path_images, segmentations_dir, global_dict_crop_centers):
    list_images = [name[:name.index('.nii')] for name in os.listdir(segmentations_dir) if name.endswith('.nii') and (name[:name.index('.nii')] not in list(global_dict_crop_centers.keys()))]

    print(len(list_images))

    # for i_name, name in enumerate(list_final) :
    for i_name,name in enumerate(list_images) :
        underscore_idxs = [i for i, ch in enumerate(name) if ch == '_']
        if 'train' in name:
            path=path_images+'train_fixed/'+name[:underscore_idxs[1]]+'/'+name[:underscore_idxs[2]]+'/'+name+'.nii.gz'
        else:
            path=path_images+'valid_fixed/'+name[:underscore_idxs[1]]+'/'+name[:underscore_idxs[2]]+'/'+name+'.nii.gz'
        # list_final[i_name] = path
        list_images[i_name] = path

        list_final = list_images

    print(len(list_final))
    return list_final

#=====================================================================================================================================================================

ImageTransforms_with_segmentations = Compose(
    [
        LoadImaged(keys=["image"], image_only=False, meta_keys=["image_meta_dict"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),

    ]
)

SegmentationTransforms = Compose(
    [
        EnsureChannelFirstd(keys=["segmentation"], channel_dim=0),
        Orientationd(keys=["segmentation"], axcodes="RAS", labels=None),
        Spacingd(keys=["segmentation"], pixdim=(1.5, 1.5, 3), mode=("nearest")),
        SpatialPadd(keys=["segmentation"], spatial_size=[224, 224, 160]),
    ]
)

class SafeCrop(InvertibleTransform, LazyTransform):
    """
    Crop around a given center, adjusting the center if the crop exceeds image bounds.
    Keeps a constant crop size without padding.
    """

    def __init__(self, roi_size: Sequence[int], lazy: bool = False):
        LazyTransform.__init__(self, lazy)
        # self.roi_center = roi_center
        self.roi_size = roi_size

    def compute_safe_slices(self, img_shape: Sequence[int], roi_center: Sequence[int]) -> tuple[slice]:
        """
        Compute slices for cropping around a center, adjusting if crop exceeds image bounds.
        """
        shape = convert_to_tensor(img_shape, dtype=torch.int16, wrap_sequence=True)
        size = convert_to_tensor(self.roi_size, dtype=torch.int16, wrap_sequence=True)

        if (roi_center is None) or np.array_equal(roi_center, np.array([-1, -1, -1])):
            center = shape // 2
        else:
            center = convert_to_tensor(roi_center, dtype=torch.int16, wrap_sequence=True)

        half = torch.div(size, 2, rounding_mode="floor") if hasattr(torch, "div") else torch.floor_divide(size, 2)
        start = center - half

        #overflow at the start
        start = torch.clamp(start, min=0)
        end = start + size

        #overflow at the end
        overflow_end = end - shape
        overflow_end = torch.clamp(overflow_end, min=0)
        start = torch.clamp(start - overflow_end, min=0)
        end = start + size


        return ensure_tuple(slice(int(s), int(e)) for s, e in zip(start, end))

    def __call__(self, img: torch.Tensor, roi_center: Sequence[int], lazy: bool | None = None) -> torch.Tensor:
        """
        Apply the safe crop transform to the image.
        """
        img_t: MetaTensor = convert_to_tensor(data=img, track_meta=True)
        spatial_shape = img_t.peek_pending_shape() if isinstance(img_t, MetaTensor) else img_t.shape[1:]
        slices = self.compute_safe_slices(spatial_shape, roi_center)

        # Add channel slice
        slices = (slice(None),) + slices
        # lazy_ = self.lazy if lazy is None else lazy
        # return crop_func(img_t, slices, lazy_, self.get_transform_info())
        return img_t[slices]

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.pop_transform(img)
        cropped = transform[TraceKeys.EXTRA_INFO]["cropped"]
        inverse_transform = BorderPad(cropped)
        with inverse_transform.trace_transform(False):
            return inverse_transform(img)


def get_joint_center(segmentation, labels):
    """
    Renvoie le centre de masse global d'un ensemble de labels.
    """
    if segmentation.ndim == 4 and segmentation.shape[0] == 1:
        segmentation = segmentation[0]
    mask = np.isin(segmentation, labels)
    if not np.any(mask):
        return np.array([-1, -1, -1])  # Aucun de ces labels n'est présent
    coords = np.argwhere(mask)
    center = coords.mean(axis=0)  # (x, y, z)
    return center  # (x, y, z)


def to_ras_nifti_from_meta(spatial_shape, affine):
    """
    Retourne un NIfTI factice en RAS et sa shape ajustée.
    """
    # Crée une image factice avec l’affine de la référence
    img_fake = nib.Nifti1Image(np.zeros(spatial_shape, dtype=np.float32), np.array(affine, dtype=np.float64))
    # Canonicalise en RAS (permutation des axes + recalcul affine)
    img_ras = nib.as_closest_canonical(img_fake)
    data_ras = img_ras.get_fdata().astype(np.float32)
    return img_ras.affine, data_ras.shape


def align_seg(
    seg_img,
    image_meta_dict,
    segmentation_path):
    
    try:
        seg_data = seg_img.get_fdata()
        seg_affine = seg_img.affine

        seg_tensor = MetaTensor(
            torch.from_numpy(seg_data).float(),  # [1, H, W, D]
            affine=torch.tensor(seg_affine)
        )
        image_tensor = MetaTensor(
            torch.from_numpy(np.zeros(image_meta_dict['spatial_shape'])).float(),  # [1, H, W, D]
            affine=torch.tensor(image_meta_dict['affine'])
        )

        data = {
            "image": image_tensor,  # dummy image
            "segmentation": seg_tensor,
        }

        resample_transform = ResampleToMatchd(
            keys="segmentation",
            key_dst="image",
            mode="nearest"
        )

        resampled_data = resample_transform(data)
        resampled_seg = resampled_data["segmentation"]
        is_LPS=0

    except Exception as e:
        # print(f"[merge_segmentations] Failed to load or resample {segmentation_path}: {e}")
        is_LPS=0
        if nib.aff2axcodes(seg_img.affine) == ('L', 'S', 'P') or nib.aff2axcodes(seg_img.affine) == ('P', 'S', 'R'):
            is_LPS = 1
            seg_img = nib.as_closest_canonical(seg_img)
            seg_data = seg_img.get_fdata()
            seg_affine = seg_img.affine

                   
            img_affine_ras, img_shape_ras = to_ras_nifti_from_meta(
                        spatial_shape=tuple(image_meta_dict["spatial_shape"]),
                        affine=image_meta_dict["affine"]
                    )
            
            seg_tensor = MetaTensor(
                torch.from_numpy(seg_data).float(),  # [1, H, W, D]
                affine=torch.tensor(seg_affine)
            )
            image_tensor = MetaTensor(
                torch.from_numpy(np.zeros(img_shape_ras)).float(),  # [1, H, W, D]
                affine=torch.tensor(img_affine_ras)
            )

            data = {
                "image": image_tensor,  # dummy image
                "segmentation": seg_tensor,
            }

            resample_transform = ResampleToMatchd(
                keys="segmentation",
                key_dst="image",
                mode="nearest"
            )

            resampled_data = resample_transform(data)
            resampled_seg = resampled_data["segmentation"]
            print(f"[merge_segmentations] Failed to load or resample {segmentation_path}: fixed")


    return resampled_seg,is_LPS

def merge_segmentations(
        resampled_seg,
        selected_names,
        image_meta_dict,
        dict_idx_segmentations, 
        is_LPS
        ):
    """
    Merge multiple binary segmentations into a single volume with integer labels,
    resampled to match the original image shape and affine.
    """
    if is_LPS == 0:
        merged = np.zeros(image_meta_dict['spatial_shape'], dtype=np.uint8)
    else:
        merged = np.zeros(resampled_seg.shape, dtype=np.uint8)

    selected_labels = [dict_idx_segmentations[name] for name in selected_names]
    mask = np.isin(resampled_seg, selected_labels)

    merged[mask] = resampled_seg[mask]

    return merged

def get_input_tensor_before_crop(reg, ct_type, resampled_seg, image_meta_dict, segmentations_for_batch_region, segmentations_to_exclude, dict_idx_segmentations, is_LPS=0):
    # Get every segmentation corresponding to TAP visible in the image 
    valid_seg_names = list(set(segmentations_for_batch_region[reg]) - set(segmentations_to_exclude[ct_type])) 


    seg_volume = merge_segmentations(
                                    resampled_seg=resampled_seg,
                                    selected_names=valid_seg_names,
                                    image_meta_dict=image_meta_dict,
                                    dict_idx_segmentations=dict_idx_segmentations,
                                    is_LPS=is_LPS
                                    )
    

    # 2. Ajouter segmentation au dictionnaire
    segmentation = seg_volume.astype(np.float32)[None, ...]

    if is_LPS == 0:
        segmentation = MetaTensor(
                        torch.from_numpy(segmentation).float(),  # [1, H, W, D]
                        affine=torch.tensor(image_meta_dict['affine'])
                    )
    else : 
        segmentation = MetaTensor(
                        torch.from_numpy(segmentation).float(),  # [1, H, W, D]
                        affine=torch.tensor(resampled_seg.affine)
                    )    
    # print("Image shape after transform:", np.shape(image_transformed))
    # print("Segmentation shape before transform:", np.shape(segmentation))

    segmentation = SegmentationTransforms({'segmentation':segmentation})["segmentation"]
    
    # print("Segmentation shape after transform:", np.shape(segmentation))

    return segmentation


def get_input_tensor_cropped(reg, image_transformed, segmentation):
    cropper = SafeCrop(roi_size=[224, 224, 160])
    # print('region : ', self.current_region)

    if segmentation is not None :
        if (reg != 'whole_report'): #(reg not in splits_to_exclude[ct_type]) and 
            crop_center = get_joint_center(segmentation=segmentation, labels=region_crop_organ_centers.get(reg))
            # print('region : ', self.current_region)
            image_cropped = cropper(image_transformed, roi_center=crop_center)
            segmentation_cropped = cropper(segmentation, roi_center=crop_center)
            # print('Crop center : ', crop_center)
        else : # if the anatomical region split is not visible on the scan we center the crop and give the whole report
            # print(f'souldnt be in the else if not whole_report : {reg}')
            image_cropped = cropper(image_transformed, roi_center=None)
            segmentation_cropped = cropper(segmentation, roi_center=None)
            # print('Crop center :', ' center')
        # print(np.shape(image_cropped))
        # print(np.shape(segmentation_cropped))

        to_tensor = ToTensor()
        image_cropped = to_tensor(image_cropped)
        segmentation_cropped = to_tensor(segmentation_cropped)
        input_tensor = torch.cat([image_cropped, segmentation_cropped], dim=0)
        return input_tensor
    else :
        image_cropped = cropper(image_transformed, roi_center=None)

        to_tensor = ToTensor()
        image_cropped = to_tensor(image_cropped)
        return image_cropped
    

#=====================================================================================================================================================================
def preprocessing_no_seg_other_datasets_get_crop_center(images, ct_type, segmentations_dir, ):
    dict_crop_centers = {}
    for i in range(len(images)):
        regions_done = []

        image_path = f'{images[i]}'
        transformed = ImageTransforms_with_segmentations({'image' : image_path})
        image_meta_dict = transformed["image_meta_dict"]
        
        key = os.path.basename(images[i])
        key = key[:key.index('.nii')]
        dict_crop_centers[key] = {}
        segmentation_path = f'{segmentations_dir}/{key}.nii'
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

                dict_crop_centers[key][reg] = crop_center
                    
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
            dict_crop_centers[key][reg] = crop_center

    return dict_crop_centers
#=====================================================================================================================================================================

def run_with_resilience(image_list, workers, ct_type, list_images, process_one_image, check_is_wrong_INSPECT=None, path_wrong_images_INSPECT=None, check_is_wrong_BIMCVR=None, path_wrong_images_BIMCVR=None): 
    # Limiter le nombre de threads des libs numériques
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    nb_wrong = 0

    # Charger les listes existantes
    if check_is_wrong_INSPECT == True:
        if os.path.isfile(path_wrong_images_INSPECT):
            list_wrong_INSPECT = list(np.unique(pickle.load(open(path_wrong_images_INSPECT, "rb"))))
        else:
            list_wrong_INSPECT = []
    else:
        list_wrong_INSPECT = []

    if check_is_wrong_BIMCVR == True:
        if os.path.isfile(path_wrong_images_BIMCVR):
            list_wrong_BIMCVR = list(np.unique(pickle.load(open(path_wrong_images_BIMCVR, "rb"))))
        else:
            list_wrong_BIMCVR = []
    else:
        list_wrong_BIMCVR = []

    # Paramètres de robustesse
    chunk_size   = 250    # traite par paquets pour limiter l'impact d'un pool cassé
    max_retries  = 2      # nombre max de retries par image
    pending      = list(image_list)
    retries      = {img: 0 for img in pending}
    processed    = 0

    while pending:
        # Prendre un paquet
        batch = pending[:chunk_size]
        pending = pending[chunk_size:]

        # Créer un pool pour ce paquet
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fut_to_img   = {executor.submit(process_one_image, img, ct_type): img for img in batch}
            pending_futs = set(fut_to_img.keys())
            pool_broken  = False

            # Traiter jusqu'à la première exception (évite de perdre du temps)
            while pending_futs:
                done, pending_futs = wait(pending_futs, return_when=FIRST_EXCEPTION)

                for fut in done:
                    img = fut_to_img[fut]
                    try:
                        image_path, is_wrong_INSPECT, is_wrong_BIMCVR = fut.result()
                        nb_wrong += is_wrong_INSPECT + is_wrong_BIMCVR

                        if (is_wrong_INSPECT == 1) and (check_is_wrong_INSPECT == True):
                            if image_path not in list_wrong_INSPECT:
                                list_wrong_INSPECT.append(image_path)

                        if (is_wrong_BIMCVR == 1) and (check_is_wrong_BIMCVR == True):
                            if image_path not in list_wrong_BIMCVR:
                                list_wrong_BIMCVR.append(image_path)

                        processed += 1
                        print(f"Done: {os.path.basename(image_path)}, number of wrong images for now : {nb_wrong}/{processed}", flush=True)

                        # Sauvegarde périodique (toutes les 50)
                        if processed % 50 == 0:
                            if check_is_wrong_INSPECT == True:
                                with open(path_wrong_images_INSPECT, "wb") as f:
                                    pickle.dump(list_wrong_INSPECT, f)
                                print(f"Total nb of wrong images : {len(list_wrong_INSPECT)}/{len(list_images)}")
                            if check_is_wrong_BIMCVR == True:
                                with open(path_wrong_images_BIMCVR, "wb") as f:
                                    pickle.dump(list_wrong_BIMCVR, f)
                                print(f"Total nb of wrong images : {len(list_wrong_BIMCVR)}/{len(list_images)}")

                    except BrokenProcessPool as e:
                        # Le pool est cassé: on arrête IMMEDIATEMENT ce batch
                        pool_broken = True
                        print(f"[POOL BROKEN] {e} – restarting executor for unfinished items")
                        # Requeue l'image courante + toutes les futures restantes
                        pending.append(img)
                        pending.extend([fut_to_img[f] for f in pending_futs])
                        break

                    except Exception as e:
                        # Erreur dans ce futur (ex: OOM transformé en MemoryError, ou autre)
                        print(f"[ERROR] {img} -> {e}")
                        if retries[img] < max_retries:
                            retries[img] += 1
                            time.sleep(0.5 * retries[img])  # petit backoff
                            pending.append(img)             # rejouer plus tard
                        else:
                            print(f"[SKIP] {img} - after {max_retries} retries")

                if pool_broken:
                    # Fermer ce pool et passer au paquet suivant (nouveau pool)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # Sauvegarde finale
    if check_is_wrong_INSPECT == True:
        with open(path_wrong_images_INSPECT, "wb") as f:
            pickle.dump(list_wrong_INSPECT, f)
    if check_is_wrong_BIMCVR == True:
        with open(path_wrong_images_BIMCVR, "wb") as f:
            pickle.dump(list_wrong_BIMCVR, f)

def run_with_resilience_no_seg(image_list, workers, ct_type, process_one_image, global_dict_crop_centers, save_global_dict_crop_centers, check_is_wrong_INSPECT=None, path_wrong_images_INSPECT=None, check_is_wrong_BIMCVR=None, path_wrong_images_BIMCVR=None): 
    # Limiter le nombre de threads des libs numériques
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    nb_wrong = 0

    # Charger les listes existantes
    if check_is_wrong_INSPECT == True:
        if os.path.isfile(path_wrong_images_INSPECT):
            list_wrong_INSPECT = list(np.unique(pickle.load(open(path_wrong_images_INSPECT, "rb"))))
        else:
            list_wrong_INSPECT = []
    else:
        list_wrong_INSPECT = []

    if check_is_wrong_BIMCVR == True:
        if os.path.isfile(path_wrong_images_BIMCVR):
            list_wrong_BIMCVR = list(np.unique(pickle.load(open(path_wrong_images_BIMCVR, "rb"))))
        else:
            list_wrong_BIMCVR = []
    else:
        list_wrong_BIMCVR = []

    # Paramètres de robustesse
    chunk_size   = 250    # traite par paquets pour limiter l'impact d'un pool cassé
    max_retries  = 2      # nombre max de retries par image
    pending      = list(image_list)
    retries      = {img: 0 for img in pending}
    processed    = 0

    while pending:
        # Prendre un paquet
        batch = pending[:chunk_size]
        pending = pending[chunk_size:]

        # Créer un pool pour ce paquet
        with ProcessPoolExecutor(max_workers=workers) as executor:
            fut_to_img   = {executor.submit(process_one_image, img, ct_type): img for img in batch}
            pending_futs = set(fut_to_img.keys())
            pool_broken  = False

            # Traiter jusqu'à la première exception (évite de perdre du temps)
            while pending_futs:
                done, pending_futs = wait(pending_futs, return_when=FIRST_EXCEPTION)

                for fut in done:
                    img = fut_to_img[fut]
                    try:
                        image_path, is_wrong_INSPECT, is_wrong_BIMCVR, dict_crop_centers = fut.result()
                        nb_wrong += is_wrong_INSPECT + is_wrong_BIMCVR
                        key = os.path.basename(image_path)
                        key = key[:key.index('.nii')]
                        global_dict_crop_centers[key] = dict_crop_centers[key]

                        if (is_wrong_INSPECT == 1) and (check_is_wrong_INSPECT == True):
                            if image_path not in list_wrong_INSPECT:
                                list_wrong_INSPECT.append(image_path)

                        if (is_wrong_BIMCVR == 1) and (check_is_wrong_BIMCVR == True):
                            if image_path not in list_wrong_BIMCVR:
                                list_wrong_BIMCVR.append(image_path)

                        processed += 1
                        # print(f"Done: {os.path.basename(image_path)}, number of wrong images for now : {nb_wrong}/{processed}", flush=True)

                        # Sauvegarde périodique (toutes les 50)
                        if processed % 10 == 0:
                            print(f"Done : {processed}/{len(image_list)}", flush=True)

                            with open(save_global_dict_crop_centers, "wb") as f:
                                pickle.dump(global_dict_crop_centers, f)
                            if check_is_wrong_INSPECT == True:
                                with open(path_wrong_images_INSPECT, "wb") as f:
                                    pickle.dump(list_wrong_INSPECT, f)
                                print(f"Total nb of wrong images : {len(list_wrong_INSPECT)}/{len(image_list)}")
                            if check_is_wrong_BIMCVR == True:
                                with open(path_wrong_images_BIMCVR, "wb") as f:
                                    pickle.dump(list_wrong_BIMCVR, f)
                                print(f"Total nb of wrong images : {len(list_wrong_BIMCVR)}/{len(image_list)}")

                    except BrokenProcessPool as e:
                        # Le pool est cassé: on arrête IMMEDIATEMENT ce batch
                        pool_broken = True
                        print(f"[POOL BROKEN] {e} – restarting executor for unfinished items")
                        # Requeue l'image courante + toutes les futures restantes
                        pending.append(img)
                        pending.extend([fut_to_img[f] for f in pending_futs])
                        break

                    except Exception as e:
                        # Erreur dans ce futur (ex: OOM transformé en MemoryError, ou autre)
                        print(f"[ERROR] {img} -> {e}")
                        if retries[img] < max_retries:
                            retries[img] += 1
                            time.sleep(0.5 * retries[img])  # petit backoff
                            pending.append(img)             # rejouer plus tard
                        else:
                            print(f"[SKIP] {img} - after {max_retries} retries")

                if pool_broken:
                    # Fermer ce pool et passer au paquet suivant (nouveau pool)
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # Sauvegarde finale

    with open(save_global_dict_crop_centers, "wb") as f:
        pickle.dump(global_dict_crop_centers, f)
    if check_is_wrong_INSPECT == True:
        with open(path_wrong_images_INSPECT, "wb") as f:
            pickle.dump(list_wrong_INSPECT, f)
    if check_is_wrong_BIMCVR == True:
        with open(path_wrong_images_BIMCVR, "wb") as f:
            pickle.dump(list_wrong_BIMCVR, f)