import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config, model_config
from models.urine_model import create_urine_model
from preprocessing.base_preprocessor import UrineDataPreprocessor


def load_and_preprocess_data(data_path, target='uti'):
    """Load and preprocess the urine dataset."""
    # Initialize preprocessor
    preprocessor = UrineDataPreprocessor(data_path)
    
    # Load data
    df = preprocessor.load_data()
    
    # Preprocess data
    X, y = preprocessor.preprocess(df, target=target)
    
    # Save preprocessor
    preprocessor.save_preprocessor(os.path.join(data_config.MODELS_DIR, 'urine_preprocessor.pkl'))
    
    # Split data
    X_train, X_val, y_train, y_val = preprocessor.split_data(
        X, y, 
        test_size=0.2, 
        random_state=model_config.SEED
    )
    
    # Print class distribution
    print(f"\nClass distribution for {target.upper()}:")
    print(f"Train - Positive: {y_train.sum()}, Negative: {len(y_train) - y_train.sum()}")
    print(f"Val - Positive: {y_val.sum()}, Negative: {len(y_val) - y_val.sum()}")
    
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=32):
    """Create PyTorch data loaders."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_val_tensor = torch.FloatTensor(X_val.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_val_tensor = torch.FloatTensor(y_val.values)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=model_config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=model_config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_model(
    target='uti',
    hidden_dims=None,
    dropout_rate=0.3,
    learning_rate=1e-4,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=100,
    patience=10,
    seed=42,
    data_path=None
):
    """Train the urine model."""
    # Set random seeds for reproducibility
    pl.seed_everything(seed)
    
    # Use default hidden_dims if not provided
    if hidden_dims is None:
        hidden_dims = model_config.HIDDEN_LAYERS
    
    # Set data path
    if data_path is None:
        data_path = data_config.RAW_DATA_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(data_config.MODELS_DIR, exist_ok=True)
    
    # Load and preprocess data
    print(f"\nLoading and preprocessing {target.upper()} data...")
    X_train, X_val, y_train, y_val = load_and_preprocess_data(data_path, target=target)
    
    # Calculate class weights for imbalanced data
    class_weights = None
    if target in ['uti', 'ckd']:  # Binary classification
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
        class_weights = torch.tensor([pos_weight], dtype=torch.float32)
        print(f"\nUsing class weight for positive class: {pos_weight:.2f}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val, batch_size=batch_size
    )
    
    # Initialize model
    model = create_urine_model(
        input_dim=X_train.shape[1],
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=data_config.MODELS_DIR,
        filename=f"{target}_model" + "_{epoch:02d}_{val_auroc:.4f}",
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
        save_dir=os.path.join("logs", target),
        name=f"{target}_logs"
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
    print(f"\nTraining {target.upper()} model...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save the best model
    best_model_path = os.path.join(data_config.MODELS_DIR, f"{target}_model.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"\nBest {target.upper()} model saved to {best_model_path}")
    
    # Return the path to the best model
    return best_model_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train urine model')
    parser.add_argument('--target', type=str, default='uti',
                       choices=['uti', 'ckd'],
                       help='Target variable to predict (default: uti)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', 
                       default=model_config.HIDDEN_LAYERS,
                       help='List of hidden layer dimensions (default: [256, 128, 64])')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs (default: 100)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to the data directory (default: data/raw)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print("\n" + "=" * 50)
    print(f"Training {args.target.upper()} model with the following configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50 + "\n")
    
    # Train the model
    best_model_path = train_model(
        target=args.target,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        data_path=args.data_path
    )
    
    print(f"\nTraining completed. Best model saved to: {best_model_path}")
