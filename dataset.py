import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import random
from torch.utils.data import Dataset
import glob 
from PIL import Image
import cv2 
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, img_dir,transform=None,name_label=None):
        
        self.path_list_class = glob.glob(img_dir + "/*")
        self.path_img = [ j for i in self.path_list_class for j in glob.glob(i + '/*')]
        self.transform = transform
        if name_label == None:
            self.name_label = [i.split('/')[-1] for i in self.path_list_class ]
            self.name_label.sort()
        else:
            self.name_label = name_label
        self.classes = len(self.name_label)
    def __len__(self):
        return len(self.path_img)

    def __getitem__(self, idx):
        label = self.name_label.index(self.path_img[idx].split('/')[-2])
        try:
            image = cv2.imread(self.path_img[idx], 1)
        
            if self.transform:
                image = self.transform(image=image)["image"]
            
        except:
            print(self.path_img[idx])
        return image, label, self.path_img[idx]


def get_data_loader(input_path,batch_size_train=32,batch_size_test=16,name_label=None):


    data_transforms = {
        'train':
                A.Compose(
            [
                A.OneOf([
                A.Affine(scale=None, rotate=(-30, 30), shear=(-16, 16), fit_output = True, p=0.5),
                ], p=0.5),
                A.Flip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
                    A.Compose([
                        A.CLAHE(p=1),
                        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1)]),
                ], p=0.5),
                A.Blur(blur_limit=(10, 10), p=0.3),
                A.Resize(224, 224, interpolation=1, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                ToTensorV2()
            ]),


        'validation':
                A.Compose(
            [
                A.Resize(224, 224, interpolation=1, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
                ToTensorV2()
            ]),


    }

    image_datasets = {
        'train': 
        CustomDataset(input_path + 'train', data_transforms['train'],name_label),
        'validation': 
        CustomDataset(input_path + 'val', data_transforms['validation'],name_label),
    }
    
    size_train = len(image_datasets['train'])
    size_val = len(image_datasets['validation'])
    print(f'[INFO]: Number of training examples: { size_train }')
    print(f'[INFO]: Number of validation examples: {size_val}')

    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=batch_size_train,
                                    shuffle=True,
                                    num_workers=4), 
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=batch_size_test,
                                    shuffle=False,
                                    num_workers=4),
                 
    }
    return dataloaders