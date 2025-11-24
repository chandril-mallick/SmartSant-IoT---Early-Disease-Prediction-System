import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import pytorch_lightning as pl
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

class UrineModel(pl.LightningModule):
    """PyTorch Lightning module for urine test classification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_classes: int = 1,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Initialize the urine model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            num_classes: Number of output classes (1 for binary)
            class_weights: Class weights for imbalanced data
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights)
        
        # Metrics
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')
        
        self.train_precision = Precision(task='binary')
        self.val_precision = Precision(task='binary')
        
        self.train_recall = Recall(task='binary')
        self.val_recall = Recall(task='binary')
        
        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')
        
        self.train_auroc = AUROC(task='binary')
        self.val_auroc = AUROC(task='binary')
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        logits = self.classifier(features)
        return logits.squeeze()
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        optimizer = Adam(
            self.parameters(),
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
        loss = self.loss_fn(logits, y.float())
        
        # Update metrics
        preds = torch.sigmoid(logits)
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
        loss = self.loss_fn(logits, y.float())
        
        # Update metrics
        preds = torch.sigmoid(logits)
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
        loss = self.loss_fn(logits, y.float())
        
        # Update metrics
        preds = torch.sigmoid(logits)
        self.test_accuracy(preds, y)
        
        # Log metrics
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return {'test_loss': loss, 'test_acc': self.test_accuracy}
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        logits = self(x)
        return torch.sigmoid(logits)


def create_urine_model(
    input_dim: int,
    target: str = 'uti',
    hidden_dims: List[int] = [256, 128, 64],
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    class_weights: Optional[torch.Tensor] = None
) -> UrineModel:
    """Create a urine model with the specified configuration."""
    if target not in ['uti', 'ckd']:
        raise ValueError("Target must be either 'uti' or 'ckd'")
    
    return UrineModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_classes=1,  # Binary classification
        class_weights=class_weights
    )
