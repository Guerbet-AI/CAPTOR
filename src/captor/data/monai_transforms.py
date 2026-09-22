from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


ImageTransforms = Compose(
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
        LoadImaged(keys=["image"], image_only=False, meta_keys=["image_meta_dict"]),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Orientationd(keys=["image"], axcodes="RAS", labels=None),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("nearest")),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
    ]
)