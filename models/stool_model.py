import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from torchvision import models
from typing import Optional, List, Dict, Any
import timm

class StoolModel(pl.LightningModule):
    """PyTorch Lightning module for stool image classification."""
    
    def __init__(
        self,
        model_name: str = 'efficientnet_b0',
        num_classes: int = 7,  # Bristol stool scale 1-7
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        freeze_backbone: bool = False
    ):
        """Initialize the stool model.
        
        Args:
            model_name: Name of the CNN backbone (from timm or torchvision)
            num_classes: Number of output classes (7 for Bristol stool scale)
            pretrained: Whether to use pretrained weights
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            class_weights: Class weights for imbalanced data
            freeze_backbone: Whether to freeze the backbone during training
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        
        # Initialize model
        self.model = self._create_model(model_name, num_classes, pretrained)
        
        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze classifier/head
            if hasattr(self.model, 'classifier'):
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(self.model, 'fc'):
                for param in self.model.fc.parameters():
                    param.requires_grad = True
        
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
    
    def _create_model(self, model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
        """Create the CNN model with the specified backbone."""
        # Try to load from timm first
        try:
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            return model
        except RuntimeError:
            # Fall back to torchvision models
            if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
                model = getattr(models, model_name)(pretrained=pretrained)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
                return model
            elif model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
                model = getattr(models, model_name)(pretrained=pretrained)
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)
                return model
            elif model_name in ['densenet121', 'densenet169', 'densenet201']:
                model = getattr(models, model_name)(pretrained=pretrained)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
                return model
            else:
                raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
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
            'monitor': 'val_auroc',
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
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
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'val_loss': loss, 'val_auroc': self.val_auroc}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Update metrics
        preds = F.softmax(logits, dim=1)
        self.test_accuracy(preds, y)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'test_loss': loss, 'test_acc': self.test_accuracy}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        return F.softmax(logits, dim=1)
    
    def get_feature_maps(self, x):
        """Extract feature maps for Grad-CAM visualization."""
        # This is a simplified version - actual implementation depends on the backbone
        features = []
        
        # For ResNet
        if hasattr(self.model, 'layer1'):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            x1 = self.model.layer1(x)
            x2 = self.model.layer2(x1)
            x3 = self.model.layer3(x2)
            x4 = self.model.layer4(x3)
            
            features = [x1, x2, x3, x4]
        # For EfficientNet
        elif hasattr(self.model, 'features'):
            features = []
            x = self.model.features[0](x)
            for i, layer in enumerate(self.model.features[1:]):
                x = layer(x)
                if i in [1, 3, 5, 7]:  # Adjust indices based on the specific model
                    features.append(x)
        
        return features


def create_stool_model(
    model_name: str = 'efficientnet_b0',
    num_classes: int = 7,
    pretrained: bool = True,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None,
    freeze_backbone: bool = True
) -> StoolModel:
    """Create a stool classification model with the specified configuration."""
    return StoolModel(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=class_weights,
        freeze_backbone=freeze_backbone
    )
