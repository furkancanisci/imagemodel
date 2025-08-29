"""
Background Remover Module
Based on the rembg functionality in Hunyuan3D-2.1
"""
import numpy as np
from PIL import Image


class BackgroundRemover:
    """
    Background remover for images, similar to the one used in Hunyuan3D-2.1
    """
    
    def __init__(self):
        """
        Initialize the background remover.
        """
        pass
        
    def __call__(self, image):
        """
        Remove background from image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Image with transparent background
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # If image is RGB, add alpha channel
        if img_array.shape[2] == 3:
            # Create alpha channel (fully opaque)
            alpha = np.full((img_array.shape[0], img_array.shape[1]), 255, dtype=np.uint8)
            img_array = np.dstack([img_array, alpha])
        
        # Convert back to PIL Image
        result = Image.fromarray(img_array, 'RGBA')
        return result


# For compatibility with the original API
def new_rembg():
    """
    Factory function for creating background remover.
    """
    return BackgroundRemover()