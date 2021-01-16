import cv2 
import albumentations as A

transforms_train = A.Compose([                       
    A.Resize(height=256, width=256, p=1.0),
    A.ShiftScaleRotate(p=0.5),
    A.Flip(),
    A.RandomBrightnessContrast()
])

transforms_valid = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
])