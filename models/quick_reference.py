"""
Quick Reference: Using Stool Image CNN Models
Demonstrates common use cases and patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.feature_extraction_demo import StoolFeatureExtractor


def example_1_basic_classification():
    """Example 1: Basic image classification"""
    print("\n" + "="*60)
    print("Example 1: Basic Image Classification")
    print("="*60)
    
    # Create model
    model = StoolFeatureExtractor(
        model_name='efficientnet_b0',
        num_classes=7,
        pretrained=True,
        freeze_backbone=True
    )
    
    # Dummy input (replace with real images)
    images = torch.randn(4, 3, 224, 224)
    
    # Get predictions
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1)
    
    print(f"\nInput: {images.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Probabilities: {probs.shape}")
    print(f"Predictions: {predictions}")
    print(f"Predicted classes: {predictions + 1}")  # Bristol scale 1-7


def example_2_feature_extraction():
    """Example 2: Feature vector extraction"""
    print("\n" + "="*60)
    print("Example 2: Feature Vector Extraction")
    print("="*60)
    
    # Create model
    model = StoolFeatureExtractor('efficientnet_b0')
    
    # Extract features
    images = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        features = model.extract_features(images)
    
    print(f"\nInput images: {images.shape}")
    print(f"Feature vectors: {features.shape}")
    print(f"Feature dimension: {model.feature_dim}")
    
    # Use cases
    print(f"\n💡 Use cases for features:")
    print(f"   - Similarity search")
    print(f"   - Clustering")
    print(f"   - Dimensionality reduction (t-SNE, UMAP)")
    print(f"   - Transfer to other tasks")


def example_3_compare_backbones():
    """Example 3: Compare different backbones"""
    print("\n" + "="*60)
    print("Example 3: Compare Different Backbones")
    print("="*60)
    
    backbones = ['efficientnet_b0', 'resnet50']
    images = torch.randn(2, 3, 224, 224)
    
    for backbone_name in backbones:
        print(f"\n{'-'*60}")
        print(f"Backbone: {backbone_name}")
        
        model = StoolFeatureExtractor(backbone_name)
        
        with torch.no_grad():
            features = model.extract_features(images)
            logits = model(images)
        
        info = model.get_model_info()
        
        print(f"  Feature dim: {features.shape[1]}")
        print(f"  Total params: {info['total_params']:,}")
        print(f"  Trainable params: {info['trainable_params']:,}")
        print(f"  Memory (approx): {info['total_params'] * 4 / 1024 / 1024:.1f} MB")


def example_4_batch_processing():
    """Example 4: Process multiple batches"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    model = StoolFeatureExtractor('efficientnet_b0')
    
    # Simulate processing multiple batches
    all_features = []
    all_predictions = []
    
    for batch_idx in range(3):
        # Dummy batch (replace with real DataLoader)
        batch = torch.randn(4, 3, 224, 224)
        
        with torch.no_grad():
            features = model.extract_features(batch)
            logits = model(batch)
            predictions = torch.argmax(logits, dim=1)
        
        all_features.append(features)
        all_predictions.append(predictions)
        
        print(f"Batch {batch_idx + 1}: {features.shape}")
    
    # Concatenate all results
    all_features = torch.cat(all_features, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    print(f"\nTotal features collected: {all_features.shape}")
    print(f"Total predictions: {all_predictions.shape}")


def example_5_model_information():
    """Example 5: Get detailed model information"""
    print("\n" + "="*60)
    print("Example 5: Model Information")
    print("="*60)
    
    model = StoolFeatureExtractor('efficientnet_b0')
    info = model.get_model_info()
    
    print(f"\n📊 Model Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Check what layers are trainable
    print(f"\n🔓 Trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"   {name}: {param.shape}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STOOL IMAGE CNN - QUICK REFERENCE EXAMPLES")
    print("="*60)
    
    # Run examples
    example_1_basic_classification()
    example_2_feature_extraction()
    example_3_compare_backbones()
    example_4_batch_processing()
    example_5_model_information()
    
    print("\n" + "="*60)
    print("✅ ALL EXAMPLES COMPLETE")
    print("="*60)
    
    print(f"\n📝 Key Takeaways:")
    print(f"   1. Use efficientnet_b0 for small datasets")
    print(f"   2. Always use pretrained=True")
    print(f"   3. Keep freeze_backbone=True to prevent overfitting")
    print(f"   4. Extract features for similarity/clustering tasks")
    print(f"   5. Process in batches for large datasets")
