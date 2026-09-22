import numpy as np
import torch
import monai
from copy import deepcopy
from typing import List, Sequence
import os
from pathlib import Path

from captor.data.monai_transforms import ImageTransforms

from monai.utils import look_up_option
from monai.data.utils import SUPPORTED_PICKLE_MOD
from monai.transforms import InvertibleTransform, LazyTransform, ToTensor, BorderPad
from monai.data.meta_tensor import MetaTensor
from monai.transforms.utils import convert_to_tensor, ensure_tuple
from monai.utils.enums import TraceKeys

regions = [
    "chest",
    "liver and biliary tree",
    "gallbladder",
    "spleen",
    "pancreas",
    "adrenal glands",
    "kidneys and ureters",
    "gastrointestinal tract",
    "peritoneal cavity",
    "pelvic organs",
    "vasculature and lymph nodes",
    "musculoskeletal"
]



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

        if roi_center is None or np.array_equal(roi_center, np.array([-1, -1, -1])):
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
        

class CTPersistentDataset(monai.data.PersistentDataset):
    def __init__(self, data, transform_image, global_dict_crop_centers, images_dir, cache_dir=None, force_write_cache=False, extension='.nii.gz'):
        super().__init__(data=data, transform=transform_image, cache_dir=cache_dir)
        self.global_dict_crop_centers = global_dict_crop_centers
        self.images_dir = images_dir
        
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.force_write_cache = force_write_cache
        
        self.current_region = 'chest'
        self.cropper = SafeCrop(roi_size=[224, 224, 160])
        self.to_tensor = ToTensor()
        self.extension = extension
        

        print(f"Size of dataset: {self.__len__()}\n")


    def _cachecheck(self, item_transformed):
        hashfile = None
        _item_transformed = deepcopy(item_transformed)

        image_data = {"image": item_transformed.get("image")}

        if self.cache_dir is not None and image_data is not None:
            data_item_md5 = self.hash_func(image_data).decode("utf-8")
            hashfile = self.cache_dir / f"{data_item_md5}.pt"

        # ---- 1) Try read cache safely ----
        if not self.force_write_cache and hashfile is not None and hashfile.is_file():
            try:
                cached_item = torch.load(hashfile, weights_only=False, map_location="cpu")
                _item_transformed["image"] = cached_item["image"]
                _item_transformed["image_meta_dict"] = cached_item["image_meta_dict"]
                return _item_transformed

            except Exception as e:
                print(f"[CACHE ERROR] Failed to load cache: {hashfile}", flush=True)
                print(f"[CACHE ERROR] Exception: {repr(e)}", flush=True)

                # Optionnel: log info
                try:
                    print(f"[CACHE ERROR] File size: {hashfile.stat().st_size} bytes", flush=True)
                except Exception:
                    pass

                # Delete corrupted cache
                try:
                    hashfile.unlink()
                    print(f"[CACHE ERROR] Deleted corrupted cache: {hashfile}", flush=True)
                except Exception as e2:
                    print(f"[CACHE ERROR] Could not delete {hashfile}: {repr(e2)}", flush=True)

                # Continue without cache

        # ---- 2) Compute preprocessing ----
        _pre_transform = self._pre_transform(image_data)
        _image_transformed = _pre_transform["image"]
        _image_meta_dict = _pre_transform["image_meta_dict"]

        _item_transformed["image"] = _image_transformed
        _item_transformed["image_meta_dict"] = _image_meta_dict

        # ---- 3) Write cache atomically ----
        if hashfile is None:
            return _item_transformed

        try:
            # Write to tmp file in SAME filesystem, then atomic replace
            tmpfile = hashfile.with_suffix(".tmp")

            torch.save(
                obj={"image": _image_transformed, "image_meta_dict": _image_meta_dict},
                f=tmpfile,
                pickle_module=look_up_option(self.pickle_module, SUPPORTED_PICKLE_MOD),
                pickle_protocol=self.pickle_protocol,
            )

            # Atomic replace (safe even if target exists)
            os.replace(tmpfile, hashfile)

        except Exception as e:
            print(f"[CACHE ERROR] Failed to write cache: {hashfile} ({repr(e)})", flush=True)
            # Best effort cleanup
            try:
                if tmpfile.exists():
                    tmpfile.unlink()
            except Exception:
                pass

        return _item_transformed

    
    def set_region(self, region: str):
        self.current_region = region

    def _transform(self, index: int):
        pre_random_item = deepcopy(self.data[index])
        vol_id = pre_random_item['image'][self.current_region]
        pre_random_item['image'] = os.path.join(self.images_dir, vol_id) + self.extension
        pre_random_item = self._cachecheck(pre_random_item)
        
        pre_random_item['crop_centers'] = self.global_dict_crop_centers[vol_id]
        for reg in regions:
            if reg not in pre_random_item['crop_centers'] :
                pre_random_item['crop_centers'][reg] = np.array([-1, -1, -1])
        item = self._post_transform(pre_random_item)

        crop_centers = item.get("crop_centers")
        if isinstance(crop_centers, dict) and self.current_region in crop_centers:
            crop_center = crop_centers[self.current_region]
        else:
            crop_center = np.array([-1, -1, -1])


        item["image"] = self.cropper(item["image"], roi_center=crop_center)

        item["image"] = self.to_tensor(item["image"])

        return item

class DataLoader(monai.data.DataLoader):
    def __init__(
        self,
        datalist: List[dict],
        images_dir: str,
        cache_dir: str,
        batchsize: int,
        global_dict_crop_centers :dict,
        shuffle: bool = True,
        num_workers: int = 0,
        force_write_cache : bool = False,
        extension : str = '.nii.gz',
        transform_image : monai.transforms.Compose = ImageTransforms

    ):
        self.datalist = datalist
        self.cache_dir = cache_dir
        self.batchsize = batchsize
        self.dataset = CTPersistentDataset(
            data=datalist,
            images_dir=images_dir,
            transform_image=transform_image,
            global_dict_crop_centers = global_dict_crop_centers,
            cache_dir=cache_dir,
            force_write_cache=force_write_cache,
            extension = extension
        )
        super().__init__(
            self.dataset,
            batch_size=batchsize,
            shuffle=shuffle,
            num_workers=num_workers,
        )