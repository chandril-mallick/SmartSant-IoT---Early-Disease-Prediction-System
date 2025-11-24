"""
Visualize Stool Image Augmentation
Shows original image vs augmented versions
"""

import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.stool_image_preprocessor import StoolImagePreprocessor
from config import data_config


def visualize_augmentation(image_path, num_augmentations=8):
    """
    Visualize original image and multiple augmented versions.
    
    Args:
        image_path: Path to the image file
        num_augmentations: Number of augmented versions to show
    """
    # Initialize preprocessor
    preprocessor = StoolImagePreprocessor()
    
    # Get transforms
    train_transform = preprocessor.get_train_transforms()
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'Image Augmentation Examples\n{os.path.basename(image_path)}', 
                 fontsize=14, fontweight='bold')
    
    # Show original
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Show augmented versions
    for idx in range(1, num_augmentations+ 1):
        row = idx // 3
        col = idx % 3
        
        # Apply augmentation
        augmented = train_transform(original_image)
        
        # Convert tensor to image for display
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        augmented = augmented * std + mean
        augmented = torch.clamp(augmented, 0, 1)
        
        # Convert to numpy
        augmented_np = augmented.permute(1, 2, 0).numpy()
        
        # Display
        axes[row, col].imshow(augmented_np)
        axes[row, col].set_title(f'Augmented #{idx}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(data_config.MODELS_DIR, 'visualization')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'augmentation_examples.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    # Find a sample image
    stool_dir = os.path.join(data_config.RAW_DATA_DIR, data_config.STOOL_IMAGES_DIR)
    
    # Get first available image
    for class_dir in sorted(Path(stool_dir).iterdir()):
        if class_dir.is_dir():
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                    print(f"Creating augmentation visualization for: {img_path.name}")
                    visualize_augmentation(str(img_path))
                    break
            break
    
    print("\n🎨 Augmentation visualization complete!")
