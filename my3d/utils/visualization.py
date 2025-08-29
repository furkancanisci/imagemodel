"""
Utility functions for image preprocessing and visualization
"""
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess an image for model input.
    
    Args:
        image_path (str): Path to the input image
        target_size (tuple): Target size for the image
        
    Returns:
        np.ndarray: Preprocessed image array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0
    
    return img_array


def visualize_depth_map(depth_map, title="Depth Map"):
    """
    Visualize a depth map.
    
    Args:
        depth_map (np.ndarray): Depth map to visualize
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()


def visualize_normal_map(normal_map, title="Normal Map"):
    """
    Visualize a normal map.
    
    Args:
        normal_map (np.ndarray): Normal map to visualize
        title (str): Title for the plot
    """
    # Normalize normals to [0, 1] for visualization
    normal_vis = (normal_map + 1.0) / 2.0
    
    plt.figure(figsize=(8, 6))
    plt.imshow(normal_vis)
    plt.title(title)
    plt.show()


def save_visualization(data, output_path, colormap='viridis'):
    """
    Save a visualization of data as an image.
    
    Args:
        data (np.ndarray): Data to visualize
        output_path (str): Path to save the visualization
        colormap (str): Colormap to use
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=colormap)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()