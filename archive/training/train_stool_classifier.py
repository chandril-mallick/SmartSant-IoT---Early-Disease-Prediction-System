"""
Bristol Stool Scale Classifier Training Script
EfficientNet-B0 with Fine-tuning and Strong Augmentations
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.append(str(Path(__file__).parent.parent))
import config

class BristolStoolDataset(Dataset):
    """Dataset for Bristol Stool Scale images with augmentation"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(img_size=224, is_training=True):
    """Get strong augmentation transforms"""
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def load_dataset(data_dir, test_size=0.2, val_size=0.1):
    """Load and split dataset"""
    
    print("\n" + "="*70)
    print("LOADING BRISTOL STOOL SCALE DATASET")
    print("="*70)
    
    image_paths = []
    labels = []
    class_counts = {}
    
    # Load images from each class folder (1-7)
    for class_idx in range(1, 8):
        class_dir = os.path.join(data_dir, str(class_idx))
        if not os.path.exists(class_dir):
            print(f"⚠️  Warning: Class {class_idx} directory not found")
            continue
        
        class_images = [
            os.path.join(class_dir, f) 
            for f in os.listdir(class_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        image_paths.extend(class_images)
        labels.extend([class_idx - 1] * len(class_images))  # 0-indexed
        class_counts[class_idx] = len(class_images)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total images: {len(image_paths)}")
    print(f"\n  Class distribution:")
    for cls, count in sorted(class_counts.items()):
        print(f"    Type {cls}: {count} images")
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )
    
    print(f"\n✂️  Data Split:")
    print(f"  Train: {len(X_train)} images")
    print(f"  Val:   {len(X_val)} images")
    print(f"  Test:  {len(X_test)} images")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 classifier for Bristol Stool Scale"""
    
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        
        # Get number of features
        num_features = self.backbone.classifier.in_features
        
        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, epoch=None):
    """Validate model"""
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    desc = f"Epoch {epoch+1} [Val]" if epoch is not None else "Validation"
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

def plot_training_history(history, save_path):
    """Plot training curves"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training history to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Type {i+1}' for i in range(7)],
                yticklabels=[f'Type {i+1}' for i in range(7)])
    plt.title('Bristol Stool Scale - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrix to {save_path}")

def main():
    # Configuration
    IMG_SIZE = 224
    BATCH_SIZE = 16  # Adjust based on GPU memory
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 7
    
    # Paths
    data_dir = os.path.join(config.data_config.RAW_DATA_DIR, 'stool_images')
    save_dir = os.path.join(config.data_config.MODELS_DIR, 'stool_classifier')
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load dataset
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(data_dir)
    
    # Create datasets
    train_transform = get_transforms(IMG_SIZE, is_training=True)
    val_transform = get_transforms(IMG_SIZE, is_training=False)
    
    train_dataset = BristolStoolDataset(X_train, y_train, train_transform)
    val_dataset = BristolStoolDataset(X_val, y_val, val_transform)
    test_dataset = BristolStoolDataset(X_test, y_test, val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Create model
    print("\n🏗️  Building EfficientNet-B0 model...")
    model = EfficientNetClassifier(num_classes=NUM_CLASSES, pretrained=True, dropout=0.3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    # Load best model for evaluation
    print("\n" + "="*70)
    print("EVALUATING BEST MODEL ON TEST SET")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc, test_preds, test_labels, test_probs = validate(
        model, test_loader, criterion, device
    )
    
    # Calculate detailed metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, test_preds, average='macro', zero_division=0
    )
    
    print(f"\n📊 Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(
        test_labels, test_preds, 
        target_names=[f'Type {i+1}' for i in range(7)],
        zero_division=0
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Save results
    results = {
        'model': 'EfficientNet-B0',
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history['train_loss']),
        'best_val_acc': float(best_val_acc),
        'test_metrics': {
            'accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        }
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to {save_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
