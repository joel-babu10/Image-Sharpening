import albumentations as A
from albumentations.pytorch import ToTensorV2

def DeblurAugmentation(image_size=256):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2(),
    ], additional_targets={'mask': 'image'})
