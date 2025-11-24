"""
Stool Image Data Preprocessing Pipeline
Handles image quality assessment, resizing, normalization, augmentation, and dataset splitting.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config, model_config


class ImageQualityAssessment:
    """Assess and filter low-quality images."""
    
    def __init__(self, min_ssim=0.3, min_size=(100, 100), max_blur_score=100):
        """
        Initialize quality assessment parameters.
        
        Args:
            min_ssim: Minimum SSIM score threshold
            min_size: Minimum image dimensions (width, height)
            max_blur_score: Maximum variance of Laplacian (higher = more blur)
        """
        self.min_ssim = min_ssim
        self.min_size = min_size
        self.max_blur_score = max_blur_score
    
    def compute_blur_score(self, image_path: str) -> float:
        """
        Compute blur score using Laplacian variance.
        Lower score = more blurry.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def compute_ssim_with_reference(self, image_path: str, reference_images: List[str]) -> float:
        """
        Compute simple similarity score with reference images.
        Uses normalized correlation instead of SSIM.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0.0
        
        image = cv2.resize(image, (224, 224))
        similarity_scores = []
        
        for ref_path in reference_images[:3]:  # Compare with top 3 references
            if ref_path == image_path:
                continue
            ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            if ref is None:
                continue
            ref = cv2.resize(ref, (224, 224))
            
            # Use template matching as similarity measure
            result = cv2.matchTemplate(image, ref, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            similarity_scores.append(score)
        
        return np.mean(similarity_scores) if similarity_scores else 0.5
    
    def assess_image_quality(self, image_path: str) -> Dict[str, float]:
        """
        Comprehensive quality assessment.
        
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Load image
            img = Image.open(image_path)
            width, height = img.size
            
            # Size check
            size_check = width >= self.min_size[0] and height >= self.min_size[1]
            
            # Blur check
            blur_score = self.compute_blur_score(image_path)
            blur_check = blur_score > self.max_blur_score
            
            # Overall quality
            quality_score = blur_score / 1000.0  # Normalize
            
            return {
                'width': width,
                'height': height,
                'blur_score': blur_score,
                'quality_score': quality_score,
                'size_check': size_check,
                'blur_check': blur_check,
                'overall_pass': size_check and blur_check
            }
        except Exception as e:
            print(f"Error assessing {image_path}: {e}")
            return {
                'width': 0,
                'height': 0,
                'blur_score': 0.0,
                'quality_score': 0.0,
                'size_check': False,
                'blur_check': False,
                'overall_pass': False
            }


class StoolImageDataset(Dataset):
    """Dataset class for stool images with augmentation."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
        image_size: int = 224
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Torchvision transforms to apply
            image_size: Target image size
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: resize and convert to tensor
            image = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])(image)
        
        return image, label


class StoolImagePreprocessor:
    """Complete preprocessing pipeline for stool images."""
    
    def __init__(
        self,
        data_dir: str = None,
        processed_dir: str = None,
        image_size: int = 224,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        random_state: int = 42,
        quality_threshold: float = 100.0
    ):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Directory containing raw images
            processed_dir: Directory to save processed images
            image_size: Target image size (square)
            train_val_test_split: Split ratios for train/val/test
            random_state: Random seed for reproducibility
            quality_threshold: Minimum blur score threshold
        """
        self.data_dir = data_dir or os.path.join(data_config.RAW_DATA_DIR, data_config.STOOL_IMAGES_DIR)
        self.processed_dir = processed_dir or os.path.join(data_config.PROCESSED_DATA_DIR, 'stool_images')
        self.image_size = image_size
        self.train_val_test_split = train_val_test_split
        self.random_state = random_state
        self.quality_threshold = quality_threshold
        
        # ImageNet normalization parameters
        self.mean = model_config.MEAN
        self.std = model_config.STD
        
        # Quality assessor
        self.quality_assessor = ImageQualityAssessment(
            min_size=(100, 100),
            max_blur_score=quality_threshold
        )
        
        # Create processed directory
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def get_train_transforms(self) -> transforms.Compose:
        """Get training augmentation transforms."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomRotation(degrees=15),  # Rotation
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
            transforms.RandomVerticalFlip(p=0.3),  # Vertical flip
            transforms.ColorJitter(  # Contrast and brightness jitter
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomResizedCrop(  # Random crop
                size=self.image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)  # ImageNet normalization
        ])
    
    def get_val_test_transforms(self) -> transforms.Compose:
        """Get validation/test transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def load_images_and_labels(self) -> Tuple[List[str], List[int]]:
        """
        Load all images and labels from directory structure.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        # Iterate through class directories
        for class_dir in sorted(Path(self.data_dir).iterdir()):
            if not class_dir.is_dir():
                continue
            
            # Class label is the directory name (1-7 for Bristol scale)
            try:
                label = int(class_dir.name)
            except ValueError:
                print(f"Skipping non-numeric directory: {class_dir.name}")
                continue
            
            # Load images from class directory
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    image_paths.append(str(img_path))
                    labels.append(label - 1)  # Convert to 0-indexed
        
        return image_paths, labels
    
    def filter_low_quality_images(
        self,
        image_paths: List[str],
        labels: List[int],
        save_report: bool = True
    ) -> Tuple[List[str], List[int], pd.DataFrame]:
        """
        Filter out low-quality images.
        
        Returns:
            Filtered (image_paths, labels, quality_report)
        """
        print("\nAssessing image quality...")
        quality_data = []
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths)):
            quality = self.quality_assessor.assess_image_quality(img_path)
            quality['path'] = img_path
            quality['label'] = label
            quality_data.append(quality)
        
        # Create DataFrame
        df = pd.DataFrame(quality_data)
        
        # Filter
        filtered_df = df[df['overall_pass']]
        
        print(f"\nQuality filtering results:")
        print(f"  Total images: {len(df)}")
        print(f"  Passed: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.1f}%)")
        print(f"  Removed: {len(df) - len(filtered_df)}")
        
        # Save quality report
        if save_report:
            report_path = os.path.join(self.processed_dir, 'quality_report.csv')
            df.to_csv(report_path, index=False)
            print(f"  Quality report saved to: {report_path}")
        
        return (
            filtered_df['path'].tolist(),
            filtered_df['label'].tolist(),
            df
        )
    
    def split_dataset(
        self,
        image_paths: List[str],
        labels: List[int]
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """
        Split dataset into train/val/test sets with stratification.
        
        Returns:
            Tuple of (train_paths, val_paths, test_paths, train_labels, val_labels, test_labels)
        """
        train_ratio, val_ratio, test_ratio = self.train_val_test_split
        
        # Check if we can stratify
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = counts.min()
        can_stratify = min_samples >= 2
        
        if not can_stratify:
            print(f"\n⚠️  Warning: Some classes have only {min_samples} sample(s).")
            print("   Disabling stratification to allow splitting.")
        
        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=labels if can_stratify else None
        )
        
        # Check if we can stratify val split
        unique_train_val, counts_train_val = np.unique(train_val_labels, return_counts=True)
        can_stratify_val = counts_train_val.min() >= 2
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_size,
            random_state=self.random_state,
            stratify=train_val_labels if can_stratify_val else None
        )
        
        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels
    
    def save_split_info(
        self,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str],
        train_labels: List[int],
        val_labels: List[int],
        test_labels: List[int]
    ):
        """Save dataset split information."""
        split_info = {
            'train': {
                'paths': train_paths,
                'labels': train_labels,
                'count': len(train_paths)
            },
            'val': {
                'paths': val_paths,
                'labels': val_labels,
                'count': len(val_paths)
            },
            'test': {
                'paths': test_paths,
                'labels': test_labels,
                'count': len(test_paths)
            }
        }
        
        # Save to JSON
        split_info_path = os.path.join(self.processed_dir, 'split_info.json')
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\nDataset split information saved to: {split_info_path}")
        
        # Print statistics
        print(f"\nDataset Split Summary:")
        print(f"  Train: {len(train_paths)} images")
        print(f"  Val:   {len(val_paths)} images")
        print(f"  Test:  {len(test_paths)} images")
        
        # Print class distribution
        for split_name, split_labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
            unique, counts = np.unique(split_labels, return_counts=True)
            print(f"\n{split_name} class distribution:")
            for cls, count in zip(unique, counts):
                print(f"    Class {cls+1}: {count} images")
    
    def create_dataloaders(
        self,
        train_paths: List[str],
        val_paths: List[str],
        test_paths: List[str],
        train_labels: List[int],
        val_labels: List[int],
        test_labels: List[int],
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = StoolImageDataset(
            train_paths, train_labels,
            transform=self.get_train_transforms()
        )
        
        val_dataset = StoolImageDataset(
            val_paths, val_labels,
            transform=self.get_val_test_transforms()
        )
        
        test_dataset = StoolImageDataset(
            test_paths, test_labels,
            transform=self.get_val_test_transforms()
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=model_config.NUM_WORKERS,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=model_config.NUM_WORKERS,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=model_config.NUM_WORKERS,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def process_all(self, filter_quality: bool = True, batch_size: int = 16):
        """
        Complete preprocessing pipeline.
        
        Args:
            filter_quality: Whether to filter low-quality images
            batch_size: Batch size for dataloaders
        """
        print("="*60)
        print("STOOL IMAGE PREPROCESSING PIPELINE")
        print("="*60)
        
        # Load images
        print("\n1. Loading images...")
        image_paths, labels = self.load_images_and_labels()
        print(f"   Found {len(image_paths)} images across {len(set(labels))} classes")
        
        # Filter quality
        if filter_quality:
            print("\n2. Filtering low-quality images...")
            image_paths, labels, quality_df = self.filter_low_quality_images(
                image_paths, labels
            )
        
        # Split dataset
        print("\n3. Splitting dataset...")
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = \
            self.split_dataset(image_paths, labels)
        
        # Save split info
        self.save_split_info(
            train_paths, val_paths, test_paths,
            train_labels, val_labels, test_labels
        )
        
        # Create dataloaders
        print("\n4. Creating dataloaders...")
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_paths, val_paths, test_paths,
            train_labels, val_labels, test_labels,
            batch_size=batch_size
        )
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Run preprocessing
    preprocessor = StoolImagePreprocessor(
        image_size=224,
        train_val_test_split=(0.7, 0.15, 0.15),
        quality_threshold=10.0  # Very low threshold due to small dataset (63 images)
    )
    
    train_loader, val_loader, test_loader = preprocessor.process_all(
        filter_quality=False,  # Disable quality filtering for small dataset
        batch_size=8  # Smaller batch size
    )
    
    print(f"\nDataLoaders created successfully!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
