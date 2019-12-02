from albumentations import (
    Blur,
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomBrightness,    
    RandomGamma,
    GaussianBlur,
    RandomCrop,
    RandomScale,
    PadIfNeeded,
    RGBShift,
    RandomResizedCrop
)


def seg_aug(image, mask):
    # image = RandomBrightnessContrast(p=0.8)(image)
    # aug = Compose([
    #             RandomBrightnessContrast(p=0.5),
    #             VerticalFlip(p=0.5),              
    #             RandomRotate90(p=0.5),
    #             ElasticTransform(p=0.5),
    #             Blur(p=0.05,blur_limit=2)
    #           ])

    aug = Compose([
              # VerticalFlip(p=0.5),
              HorizontalFlip(p=0.5),
              RandomRotate90(p=0.5),              
              RandomBrightnessContrast(p=0.2, brightness_limit=0.05, contrast_limit=0.05),
              GridDistortion(distort_limit=0.1,p=0.3),
            #   GaussianBlur(p=0.1),
              ])

    augmented = aug(image=image, mask=mask)
    return augmented

def crop_aug(image, mask, h, w, scale=(0.6, 1.0), ratio=(1.8, 2)):
    # image = RandomBrightnessContrast(p=0.8)(image)
    # aug = Compose([
    #             RandomBrightnessContrast(p=0.5),
    #             VerticalFlip(p=0.5),              
    #             RandomRotate90(p=0.5),
    #             ElasticTransform(p=0.5),
    #             Blur(p=0.05,blur_limit=2)
    #           ])

    aug = Compose([
              #  VerticalFlip(p=0.5),
              HorizontalFlip(p=0.5),              
              RandomBrightnessContrast(p=0.5),
            #   GaussianBlur(p=0.1),
              RandomResizedCrop(height=h, width=w, scale=scale, ratio=ratio)
              # RandomScale(scale_limit=0.2, p=1),
              # PadIfNeeded(min_width=w, min_height=h,p=1),
              # RandomCrop(width=w, height=h, p=1)
              ])

    augmented = aug(image=image, mask=mask)
    return augmented