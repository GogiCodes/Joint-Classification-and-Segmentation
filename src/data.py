"""
Data utilities for loading and preprocessing images
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class MultiTaskDataset(Dataset):
    """
    Dataset for joint classification and segmentation tasks
    
    Directory structure expected:
    data/
    ├── train/
    │   ├── images/
    │   ├── labels_cls/
    │   └── labels_seg/
    ├── val/
    └── test/
    """
    
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Args:
            root_dir (str): Root directory containing the dataset
            split (str): 'train', 'val', or 'test'
            transform (callable): Transforms to apply to images
            target_transform (callable): Transforms to apply to segmentation masks
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.cls_label_dir = os.path.join(root_dir, split, 'labels_cls')
        self.seg_label_dir = os.path.join(root_dir, split, 'labels_seg')
        
        # Get list of image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                   if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        # Load classification label
        cls_label_path = os.path.join(self.cls_label_dir, 
                                      self.image_files[idx].replace('.jpg', '.txt')
                                      .replace('.png', '.txt').replace('.jpeg', '.txt'))
        cls_label = int(open(cls_label_path).read().strip())
        
        # Load segmentation mask
        seg_mask_path = os.path.join(self.seg_label_dir, self.image_files[idx])
        seg_mask = Image.open(seg_mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            seg_mask = self.target_transform(seg_mask)
        else:
            seg_mask = torch.from_numpy(np.array(seg_mask)).long()
        
        return {
            'image': image,
            'cls_label': torch.tensor(cls_label),
            'seg_mask': seg_mask
        }


def get_transforms(img_size=224):
    """Get train and val transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_dataloaders(root_dir, batch_size=32, num_workers=4, img_size=224):
    """
    Get train, val, and test dataloaders
    
    Args:
        root_dir (str): Root directory of dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        img_size (int): Image size for resizing
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = MultiTaskDataset(root_dir, split='train', transform=train_transform)
    val_dataset = MultiTaskDataset(root_dir, split='val', transform=val_transform)
    test_dataset = MultiTaskDataset(root_dir, split='test', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
