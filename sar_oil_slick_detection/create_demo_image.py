#!/usr/bin/env python3
"""
Create a synthetic Red Sea SAR image for testing oil slick detection.
This simulates a SAR image with potential oil slicks for demo purposes.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_red_sea_demo_sar():
    """Create a synthetic SAR image simulating Red Sea waters with potential oil slicks."""
    
    # Create base image (224x224 grayscale SAR-like image)
    np.random.seed(42)  # For reproducibility
    
    # Base water return (darker areas typical of SAR water)
    base_water = np.random.normal(0.3, 0.08, (224, 224))
    
    # Add some texture variation typical of SAR
    x, y = np.meshgrid(np.linspace(0, 10, 224), np.linspace(0, 10, 224))
    texture = 0.05 * np.sin(x * 2) * np.cos(y * 1.5)
    base_water += texture
    
    # Add potential oil slick areas (darker regions)
    # Oil slick 1 - larger area
    oil_area_1 = np.zeros((224, 224))
    for i in range(80, 140):
        for j in range(60, 160):
            if ((i-110)**2 + (j-110)**2) < 30**2:  # Circular-ish oil slick
                oil_area_1[i, j] = 0.6  # Darkening factor
    
    # Oil slick 2 - smaller area
    oil_area_2 = np.zeros((224, 224))
    for i in range(150, 180):
        for j in range(140, 180):
            if ((i-165)**2 + (j-160)**2) < 15**2:  # Smaller circular oil slick
                oil_area_2[i, j] = 0.7  # Darkening factor
    
    # Apply oil darkening (oil appears darker in SAR)
    final_image = base_water * (1 - oil_area_1) * (1 - oil_area_2)
    
    # Add some speckle noise typical of SAR
    speckle = np.random.gamma(1, 0.05, (224, 224))
    final_image += speckle
    
    # Normalize to 0-1 range
    final_image = np.clip(final_image, 0, 1)
    
    # Convert to 8-bit image
    final_image = (final_image * 255).astype(np.uint8)
    
    return final_image

def save_demo_images():
    """Create and save demo SAR image."""
    
    print("Creating Red Sea demo SAR image...")
    demo_image = create_red_sea_demo_sar()
    
    # Save as PIL Image
    img = Image.fromarray(demo_image, mode='L')
    demo_path = "/Users/lakshmikotaru/Documents/sar_oil_slick_detection/red_sea_demo.png"
    img.save(demo_path)
    print(f"✅ Saved demo image: {demo_path}")
    
    # Also save a visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(demo_image, cmap='gray')
    plt.title('Red Sea Demo SAR Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(demo_image, cmap='hot')
    plt.title('Red Sea Demo SAR Image (Hot Colormap)')
    plt.axis('off')
    
    plt.tight_layout()
    viz_path = "/Users/lakshmikotaru/Documents/sar_oil_slick_detection/red_sea_demo_visualization.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization: {viz_path}")
    
    return demo_path

if __name__ == "__main__":
    save_demo_images()
