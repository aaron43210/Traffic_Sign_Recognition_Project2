"""
Data Loader Module
==================
Handles downloading the GTSRB dataset via torchvision, creating
train/validation/test splits, applying transforms, implementing
class-balanced sampling, and providing DataLoaders.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from collections import Counter

from src.config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, VAL_SPLIT,
    DATASET_MEAN, DATASET_STD, NUM_CLASSES, RANDOM_SEED
)
from src.preprocessing import CLAHETransform
from src.augmentation import get_train_transforms, get_test_transforms


def get_raw_dataset(split='train'):
    """
    Load raw GTSRB dataset without transforms (for EDA).
    
    Parameters
    ----------
    split : str
        'train' or 'test'
        
    Returns
    -------
    torchvision.datasets.GTSRB
    """
    basic_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.GTSRB(
        root=DATA_DIR,
        split=split,
        transform=basic_transform,
        download=True
    )
    
    print(f"[✓] Loaded raw GTSRB {split} set: {len(dataset)} images")
    return dataset


def get_raw_pil_dataset(split='train'):
    """
    Load raw GTSRB dataset with PIL Images only (for preprocessing visualization).
    
    Parameters
    ----------
    split : str
        'train' or 'test'
        
    Returns
    -------
    torchvision.datasets.GTSRB
    """
    dataset = datasets.GTSRB(
        root=DATA_DIR,
        split=split,
        transform=transforms.Resize((IMG_SIZE, IMG_SIZE)),
        download=True
    )
    
    print(f"[✓] Loaded raw PIL GTSRB {split} set: {len(dataset)} images")
    return dataset


def get_datasets():
    """
    Create train, validation, and test datasets with appropriate transforms.
    
    Returns
    -------
    tuple
        (train_dataset, val_dataset, test_dataset)
    """
    train_transform = get_train_transforms()
    test_transform = get_test_transforms()
    
    # Load full training set
    full_train = datasets.GTSRB(
        root=DATA_DIR,
        split='train',
        transform=None,  # We'll apply transforms via subsets
        download=True
    )
    
    # Split into train and validation
    total = len(full_train)
    val_size = int(total * VAL_SPLIT)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_indices, val_indices = random_split(
        range(total), [train_size, val_size], generator=generator
    )
    
    # Create subset datasets with different transforms
    train_dataset = TransformedSubset(full_train, train_indices.indices, train_transform)
    val_dataset = TransformedSubset(full_train, val_indices.indices, test_transform)
    
    # Test set
    test_dataset = datasets.GTSRB(
        root=DATA_DIR,
        split='test',
        transform=test_transform,
        download=True
    )
    
    print(f"[✓] Dataset splits created:")
    print(f"    Training:   {len(train_dataset):,} images")
    print(f"    Validation: {len(val_dataset):,} images")
    print(f"    Test:       {len(test_dataset):,} images")
    
    return train_dataset, val_dataset, test_dataset


class TransformedSubset(torch.utils.data.Dataset):
    """
    A dataset subset that applies a specific transform.
    This allows train/val splits to have different augmentation pipelines.
    """
    
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.indices)
    
    def get_labels(self):
        """Return all labels for this subset (for weighted sampler)."""
        labels = []
        for idx in self.indices:
            _, label = self.dataset[idx]
            labels.append(label)
        return labels


def get_class_weights(dataset):
    """
    Compute class weights inversely proportional to class frequency.
    
    Parameters
    ----------
    dataset : TransformedSubset or similar
        Dataset with get_labels() method or iterable (img, label) pairs
        
    Returns
    -------
    torch.Tensor
        Class weights tensor of shape (NUM_CLASSES,)
    """
    if hasattr(dataset, 'get_labels'):
        labels = dataset.get_labels()
    else:
        labels = [label for _, label in dataset]
    
    counter = Counter(labels)
    total = len(labels)
    
    weights = torch.zeros(NUM_CLASSES)
    for cls_id in range(NUM_CLASSES):
        count = counter.get(cls_id, 1)
        weights[cls_id] = total / (NUM_CLASSES * count)
    
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    
    print(f"[✓] Class weights computed (min={weights.min():.3f}, "
          f"max={weights.max():.3f}, mean={weights.mean():.3f})")
    
    return weights


def get_weighted_sampler(dataset):
    """
    Create a WeightedRandomSampler for class-balanced training.
    
    Parameters
    ----------
    dataset : TransformedSubset
        Training dataset with get_labels() method
        
    Returns
    -------
    WeightedRandomSampler
    """
    labels = dataset.get_labels()
    counter = Counter(labels)
    
    # Weight for each sample = 1 / class_count
    sample_weights = [1.0 / counter[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"[✓] Weighted sampler created for {len(labels)} samples across "
          f"{len(counter)} classes")
    
    return sampler


def get_dataloaders(balanced=True):
    """
    Create complete train/val/test DataLoaders.
    
    Parameters
    ----------
    balanced : bool
        If True, use WeightedRandomSampler for training
        
    Returns
    -------
    tuple
        (train_loader, val_loader, test_loader, class_weights)
    """
    train_dataset, val_dataset, test_dataset = get_datasets()
    class_weights = get_class_weights(train_dataset)
    
    if balanced:
        sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"\n[✓] DataLoaders ready:")
    print(f"    Train: {len(train_loader)} batches (batch_size={BATCH_SIZE})")
    print(f"    Val:   {len(val_loader)} batches")
    print(f"    Test:  {len(test_loader)} batches")
    print(f"    Balanced sampling: {balanced}")
    
    return train_loader, val_loader, test_loader, class_weights
