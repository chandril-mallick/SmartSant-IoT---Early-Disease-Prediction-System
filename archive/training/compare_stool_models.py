"""
Bristol Stool Scale - Multi-Model Comparison
Train and evaluate multiple architectures to find the best performer
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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
import config

class BristolStoolDataset(Dataset):
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

def load_dataset(data_dir, test_size=0.2):
    image_paths = []
    labels = []
    
    for class_idx in range(1, 8):
        class_dir = os.path.join(data_dir, str(class_idx))
        if not os.path.exists(class_dir):
            continue
        
        class_images = [
            os.path.join(class_dir, f) 
            for f in os.listdir(class_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        image_paths.extend(class_images)
        labels.extend([class_idx - 1] * len(class_images))
    
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    return X_train, X_test, y_train, y_test

def create_model(model_name, num_classes=7, dropout=0.3):
    """Create model based on architecture name"""
    
    if model_name == 'efficientnet_b0':
        backbone = timm.create_model('efficientnet_b0', pretrained=True)
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        return backbone
    
    elif model_name == 'efficientnet_b2':
        backbone = timm.create_model('efficientnet_b2', pretrained=True)
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        return backbone
    
    elif model_name == 'resnet50':
        backbone = timm.create_model('resnet50', pretrained=True)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        return backbone
    
    elif model_name == 'mobilenetv3':
        backbone = timm.create_model('mobilenetv3_large_100', pretrained=True)
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        return backbone
    
    elif model_name == 'densenet121':
        backbone = timm.create_model('densenet121', pretrained=True)
        num_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout/2),
            nn.Linear(512, num_classes)
        )
        return backbone

def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, device, epochs=25):
    """Train and evaluate a single model"""
    
    print(f"\n{'='*70}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*70}")
    
    # Create datasets
    train_transform = get_transforms(224, is_training=True)
    test_transform = get_transforms(224, is_training=False)
    
    train_dataset = BristolStoolDataset(X_train, y_train, train_transform)
    test_dataset = BristolStoolDataset(X_test, y_test, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Create model
    model = create_model(model_name)
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    patience_counter = 0
    patience = 10
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Evaluate
        model.eval()
        test_preds = []
        test_labels_list = []
        test_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                test_preds.extend(preds.cpu().numpy())
                test_labels_list.extend(labels.cpu().numpy())
                test_probs.extend(probs.cpu().numpy())
        
        test_acc = accuracy_score(test_labels_list, test_preds)
        
        scheduler.step(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_preds = test_preds
            best_probs = test_probs
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best: {best_acc:.4f}")
    
    # Calculate final metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_list, best_preds, average='macro', zero_division=0
    )
    
    # Calculate AUC
    try:
        y_test_bin = label_binarize(test_labels_list, classes=range(7))
        auc = roc_auc_score(y_test_bin, best_probs, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    results = {
        'model': model_name,
        'accuracy': float(best_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc)
    }
    
    print(f"\n✅ {model_name} Results:")
    print(f"   Accuracy:  {best_acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   AUC-ROC:   {auc:.4f}")
    
    return results

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = os.path.join(config.data_config.RAW_DATA_DIR, 'stool_images')
    save_dir = os.path.join(config.data_config.MODELS_DIR, 'stool_classifier')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    X_train, X_test, y_train, y_test = load_dataset(data_dir)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Models to compare
    models = [
        'efficientnet_b0',
        'efficientnet_b2',
        'resnet50',
        'mobilenetv3',
        'densenet121'
    ]
    
    all_results = {}
    
    # Train each model
    for model_name in models:
        try:
            results = train_and_evaluate(model_name, X_train, X_test, y_train, y_test, device, epochs=25)
            all_results[model_name] = results
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Save results
    with open(os.path.join(save_dir, 'model_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON TABLE")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
    print("-"*70)
    
    for model_name, results in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1']:<12.4f} {results['auc']:<12.4f}")
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model[0].upper()}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
