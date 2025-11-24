import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config, model_config
from models.stool_model import create_stool_model


class StoolDataset(Dataset):
    """Custom dataset for stool images."""
    
    def __init__(self, image_paths, labels=None, transform=None, is_train=True):
        """
        Args:
            image_paths: List of paths to stool images
            labels: List of corresponding labels (None for test set)
            transform: Optional transform to be applied on a sample
            is_train: Whether this is a training set (for data augmentation)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        
        # Define basic transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((model_config.IMAGE_SIZE, model_config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.MEAN, std=model_config.STD)
        ])
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(model_config.IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=model_config.MEAN, std=model_config.STD)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform is not None:
                image = self.transform(image)
            else:
                if self.is_train:
                    image = self.train_transform(image)
                else:
                    image = self.base_transform(image)
            
            # Return (image, label) if training/validation, else just image
            if self.labels is not None:
                label = self.labels[idx]
                return image, label
            return image
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a zero tensor if image loading fails
            dummy_image = torch.zeros((3, model_config.IMAGE_SIZE, model_config.IMAGE_SIZE))
            if self.labels is not None:
                return dummy_image, -1  # Use -1 as an invalid label
            return dummy_image


def load_stool_data(data_dir, test_size=0.2, random_state=42):
    """Load stool images and split into train/val sets."""
    # Get list of all image files
    image_paths = []
    labels = []
    
    # Expected directory structure: data_dir/class_1/, data_dir/class_2/, etc.
    class_dirs = sorted([d for d in os.listdir(data_dir) 
                        if os.path.isdir(os.path.join(data_dir, d))])
    
    print("\nFound classes:", class_dirs)
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Get all image files in the class directory
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                image_paths.append(img_path)
                labels.append(class_idx)  # Use folder index as label
    
    print(f"\nTotal images: {len(image_paths)}")
    print("Class distribution:", pd.Series(labels).value_counts().sort_index())
    
    # Split into train and validation sets
    if len(image_paths) > 0:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        print(f"\nTraining set: {len(train_paths)} images")
        print("Training class distribution:", pd.Series(train_labels).value_counts().sort_index())
        
        print(f"\nValidation set: {len(val_paths)} images")
        print("Validation class distribution:", pd.Series(val_labels).value_counts().sort_index())
        
        return train_paths, val_paths, train_labels, val_labels
    
    return [], [], [], []


def train_model(
    data_dir,
    model_name='efficientnet_b0',
    pretrained=True,
    freeze_backbone=True,
    learning_rate=1e-4,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=50,
    patience=10,
    num_workers=4,
    seed=42
):
    """Train the stool classification model."""
    # Set random seeds for reproducibility
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(data_config.MODELS_DIR, exist_ok=True)
    
    # Load and split data
    print("\nLoading and preprocessing stool images...")
    train_paths, val_paths, train_labels, val_labels = load_stool_data(
        data_dir, test_size=0.2, random_state=seed
    )
    
    if not train_paths:
        raise ValueError("No training images found. Please check the data directory structure.")
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print("\nClass weights:", class_weights.tolist())
    
    # Create datasets
    train_dataset = StoolDataset(train_paths, train_labels, is_train=True)
    val_dataset = StoolDataset(val_paths, val_labels, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model
    num_classes = len(np.unique(train_labels))
    model = create_stool_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        freeze_backbone=freeze_backbone
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=data_config.MODELS_DIR,
        filename=f"{model_name}_stool_model" + "_{epoch:02d}_{val_auroc:.4f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        save_weights_only=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_auroc",
        patience=patience,
        verbose=True,
        mode="max"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join("logs", "stool"),
        name=f"{model_name}_logs"
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        accelerator="auto",  # Automatically select GPU if available
        devices=1,
    )
    
    # Train the model
    print("\nTraining stool classification model...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the best model
    best_model_path = os.path.join(data_config.MODELS_DIR, data_config.STOOL_MODEL)
    torch.save(model.state_dict(), best_model_path)
    print(f"\nBest stool model saved to {best_model_path}")
    
    return best_model_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train stool classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the directory containing stool images')
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                       help='Model architecture (default: efficientnet_b0)')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                       help='Do not use pretrained weights')
    parser.add_argument('--no_freeze', action='store_false', dest='freeze_backbone',
                       help='Do not freeze the backbone during training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of epochs (default: 50)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping (default: 10)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 50)
    print("Training stool classification model with the following configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50 + "\n")
    
    # Train the model
    best_model_path = train_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"\nTraining completed. Best model saved to: {best_model_path}")
