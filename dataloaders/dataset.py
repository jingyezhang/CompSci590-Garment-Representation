import os
import os.path
import glob
import fnmatch  # pattern matching
import numpy as np
from numpy import linalg as LA
from random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
import random
from torchvision import transforms

input_options = ['t', 'v', 'gt']

height, width = 256, 256

from torchvision import transforms

# Define transforms for each image type
transform_tactile = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()
])

transform_visual = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()
])

transform_gt = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()
])

transform_mask = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.ToTensor()  # This will automatically scale pixels to [0, 1]
])

transform_normalize_grayscale = transforms.Compose([
    transforms.Resize((height, width)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


class TactileVisualDataset(data.Dataset):
    """A data loader for the tactile-visual dataset."""

    def __init__(self, split, args, root_dir='../data/train', val_split=0.05, seed=42):
        self.root_dir = root_dir
        self.split = split
        self.modality = args.modality

        # Set the random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        all_folders = os.listdir(self.root_dir)
        all_folders = [f for f in all_folders if os.path.isdir(os.path.join(self.root_dir, f))]

        # Shuffle and split the folders
        random.shuffle(all_folders)
        split_idx = int(len(all_folders) * val_split)
        train_folders = all_folders[split_idx:]
        val_folders = all_folders[:split_idx]

        self.folders = train_folders if split == 'train' else val_folders

    def __getraw__(self, index):
        folder_name = self.folders[index]
        folder_path = os.path.join(self.root_dir, folder_name)

        # Construct file paths
        tactile_path = os.path.join(folder_path, folder_name + '_t_sparse.png')
        visual_path = os.path.join(folder_path, folder_name + '_v.png')
        gt_path = os.path.join(folder_path, folder_name + '_t.png')
        mask_path = os.path.join(folder_path, folder_name + '_t_mask.png')

        # Read images
        tactile_image = Image.open(tactile_path).convert('RGB')
        visual_image = Image.open(visual_path).convert('RGB')
        ground_truth = Image.open(gt_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        return tactile_image, visual_image, ground_truth, mask

    def __getitem__(self, index):
        tactile_image, visual_image, ground_truth, mask = self.__getraw__(index)
        
        # Apply separate transforms to each image      
        visual_image = transform_visual(visual_image)
        mask = transform_mask(mask)
        
        # Check if we need to convert images to grayscale
        if self.modality == "g":
            tactile_image = transform_normalize_grayscale(tactile_image)
            ground_truth = transform_normalize_grayscale(ground_truth)
        else:
            # Apply the standard transforms if not converting to grayscale
            tactile_image = transform_tactile(tactile_image)
            ground_truth = transform_gt(ground_truth)
            
        items = {"t": tactile_image, "v": visual_image, "gt": ground_truth, "m": mask}
        return items


    def __len__(self): 
        return len(self.folders)

