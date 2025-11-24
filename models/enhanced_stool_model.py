"""
Enhanced Stool Image Model with Feature Extraction
Supports multiple CNN backbones: EfficientNet, ResNet, MobileNet, ViT
Includes feature vector extraction (256-1024 dimensions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torchvision import models
from typing import Optional, List, Dict, Any, Tuple
import timm


class FeatureExtractor(nn.Module):
    """
    Feature extractor wrapper that provides feature vectors from CNN backbones.
    Extracts features from the penultimate layer (before classification).
    """
    
    def __init__(self, backbone: nn.Module, feature_dim: int):
        """
        Initialize feature extractor.
        
        Args:
            backbone: CNN backbone model
            feature_dim: Dimension of feature vectors
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with feature extraction.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (features, logits)
            - features: Feature vectors [B, feature_dim]
            - logits: Classification logits [B, num_classes]
        """
        # Extract features from backbone
        features = self.extract_features(x)
        
        # Get classification output
        logits = self.backbone(x)
        
        return features, logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vectors from the backbone.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Feature vectors [B, feature_dim]
        """
        # This method needs to be customized based on the backbone architecture
        # For most models, we extract from the layer before classification
        
        if hasattr(self.backbone, 'forward_features'):
            # timm models have forward_features method
            features = self.backbone.forward_features(x)
            if len(features.shape) == 4:  # [B, C, H, W]
                features = self.global_pool(features).flatten(1)
        elif hasattr(self.backbone, 'features'):
            # EfficientNet, DenseNet from torchvision
            features = self.backbone.features(x)
            features = self.global_pool(features).flatten(1)
        elif hasattr(self.backbone, 'layer4'):
            # ResNet from torchvision
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)
            features = self.global_pool(features).flatten(1)
        else:
            # Fallback: try to extract from avgpool
            raise NotImplementedError(f"Feature extraction not implemented for this backbone type")
        
        return features


class EnhancedStoolModel(pl.LightningModule):
    """
    Enhanced PyTorch Lightning module for stool image classification.
    Supports multiple backbones and feature extraction.
    """
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 7,  # Bristol stool scale 1-7
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        freeze_backbone: bool = True,
        feature_dim: Optional[int] = None
    ):
        """
        Initialize the enhanced stool model.
        
        Args:
            model_name: CNN backbone name (efficientnet_b0, resnet50, mobilenetv3_small, vit_tiny_patch16_224)
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for L2 regularization
            class_weights: Class weights for imbalanced data
            freeze_backbone: Freeze early layers for transfer learning
            feature_dim: Feature vector dimension (auto-detected if None)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        self.freeze_backbone = freeze_backbone
        
        # Create backbone model
        self.backbone = self._create_backbone(model_name, num_classes, pretrained)
        
        # Determine feature dimension
        self.feature_dim = feature_dim or self._get_feature_dim()
        
        # Create feature extractor
        self.feature_extractor = FeatureExtractor(self.backbone, self.feature_dim)
        
        # Global average pooling (already in FeatureExtractor)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            self._freeze_backbone_layers()
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        
        self.train_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        self.val_precision = Precision(task='multiclass', num_classes=num_classes, average='macro')
        
        self.train_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        self.val_recall = Recall(task='multiclass', num_classes=num_classes, average='macro')
        
        self.train_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.val_f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')
        
        self.train_auroc = AUROC(task='multiclass', num_classes=num_classes, average='macro')
        self.val_auroc = AUROC(task='multiclass', num_classes=num_classes, average='macro')
    
    def _create_backbone(self, model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """
        Create CNN backbone with pretrained ImageNet weights.
        
        Supported backbones:
        - EfficientNet: efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3
        - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
        - MobileNet: mobilenetv3_small, mobilenetv3_large
        - ViT: vit_tiny_patch16_224, vit_small_patch16_224, vit_base_patch16_224
        """
        print(f"\n{'='*60}")
        print(f"Creating {model_name} backbone")
        print(f"  Pretrained: {pretrained}")
        print(f"  Num classes: {num_classes}")
        print(f"{'='*60}")
        
        try:
            # Try loading from timm (recommended for most models)
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            print(f"✅ Loaded {model_name} from timm library")
            return model
            
        except Exception as e:
            print(f"⚠️  Could not load from timm: {e}")
            print(f"Trying torchvision...")
            
            # Fallback to torchvision
            if 'resnet' in model_name:
                model = getattr(models, model_name)(weights='IMAGENET1K_V1' if pretrained else None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                print(f"✅ Loaded {model_name} from torchvision")
                return model
                
            elif 'efficientnet' in model_name:
                weights = 'IMAGENET1K_V1' if pretrained else None
                model = getattr(models, model_name)(weights=weights)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)
                print(f"✅ Loaded {model_name} from torchvision")
                return model
                
            elif 'mobilenet' in model_name:
                weights = 'IMAGENET1K_V1' if pretrained else None
                model = getattr(models, model_name)(weights=weights)
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
                print(f"✅ Loaded {model_name} from torchvision")
                return model
                
            else:
                raise ValueError(f"Unsupported model: {model_name}")
    
    def _get_feature_dim(self) -> int:
        """Auto-detect feature dimension from the backbone."""
        if hasattr(self.backbone, 'num_features'):
            return self.backbone.num_features
        elif hasattr(self.backbone, 'fc'):
            return self.backbone.fc.in_features
        elif hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                for layer in reversed(self.backbone.classifier):
                    if isinstance(layer, nn.Linear):
                        return layer.in_features
            elif isinstance(self.backbone.classifier, nn.Linear):
                return self.backbone.classifier.in_features
        
        # Default for common backbones
        model_dims = {
            'efficientnet_b0': 1280,
            'efficientnet_b1': 1280,
            'efficientnet_b2': 1408,
            'efficientnet_b3': 1536,
            'resnet18': 512,
            'resnet34': 512,
            'resnet50': 2048,
            'resnet101': 2048,
            'resnet152': 2048,
            'mobilenetv3_small': 576,
            'mobilenetv3_large': 960,
            'vit_tiny_patch16_224': 192,
            'vit_small_patch16_224': 384,
            'vit_base_patch16_224': 768,
        }
        
        return model_dims.get(self.model_name, 512)
    
    def _freeze_backbone_layers(self):
        """Freeze early layers for transfer learning."""
        print(f"\n🔒 Freezing backbone layers for transfer learning...")
        
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier/head only
        if hasattr(self.backbone, 'classifier'):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
            print(f"   ✅ Unfroze classifier layer")
            
        elif hasattr(self.backbone, 'fc'):
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            print(f"   ✅ Unfroze fc layer")
            
        elif hasattr(self.backbone, 'head'):
            for param in self.backbone.head.parameters():
                param.requires_grad = True
            print(f"   ✅ Unfroze head layer")
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"   Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification."""
        return self.backbone(x)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vectors from input images.
        
        Args:
            x: Input images [B, C, H, W]
            
        Returns:
            Feature vectors [B, feature_dim]
        """
        return self.feature_extractor.extract_features(x)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Only optimize parameters that require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = Adam(
            params_to_optimize,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            ),
            'monitor': 'val_acc',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Update metrics
        preds = F.softmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)
        self.train_auroc(preds, y)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Update metrics
        preds = F.softmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_f1(preds, y)
        self.val_auroc(preds, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Update metrics
        preds = F.softmax(logits, dim=1)
        self.test_accuracy(preds, y)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'test_loss': loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Make predictions with probabilities."""
        x, _ = batch
        logits = self(x)
        return F.softmax(logits, dim=1)


def create_stool_model(
    model_name: str = 'efficientnet_b0',
    num_classes: int = 7,
    pretrained: bool = True,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    freeze_backbone: bool = True
) -> EnhancedStoolModel:
    """
    Create a stool classification model with specified configuration.
    
    Supported backbones:
    - EfficientNet: efficientnet_b0 (1280-D), efficientnet_b1 (1280-D), efficientnet_b2 (1408-D)
    - ResNet: resnet18 (512-D), resnet34 (512-D), resnet50 (2048-D)
    - MobileNet: mobilenetv3_small (576-D), mobilenetv3_large (960-D)
    - ViT: vit_tiny_patch16_224 (192-D), vit_small_patch16_224 (384-D)
    
    Args:
        model_name: Backbone architecture name
        num_classes: Number of output classes (7 for Bristol scale)
        pretrained: Use ImageNet pretrained weights
        learning_rate: Learning rate
        weight_decay: Weight decay
        class_weights: Class weights for imbalanced data
        freeze_backbone: Freeze early layers
        
    Returns:
        EnhancedStoolModel instance
    """
    return EnhancedStoolModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        freeze_backbone=freeze_backbone
    )


if __name__ == "__main__":
    # Test model creation with different backbones
    print("\n" + "="*60)
    print("TESTING CNN BACKBONES")
    print("="*60)
    
    backbones = [
        ('efficientnet_b0', 1280),
        ('resnet50', 2048),
        ('mobilenetv3_small', 576),
    ]
    
    for model_name, expected_dim in backbones:
        print(f"\n{'-'*60}")
        try:
            model = create_stool_model(
                model_name=model_name,
                pretrained=True,
                freeze_backbone=True
            )
            
            # Test feature extraction
            dummy_input = torch.randn(2, 3, 224, 224)
            features = model.extract_features(dummy_input)
            
            print(f"✅ {model_name}:")
            print(f"   Feature dim: {model.feature_dim}")
            print(f"   Extracted shape: {features.shape}")
            print(f"   Expected: ({2}, {expected_dim})")
            print(f"   Match: {'✅' if features.shape[1] == expected_dim else '❌'}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
    
    print(f"\n{'='*60}")
    print("BACKBONE TESTING COMPLETE")
    print("="*60)
