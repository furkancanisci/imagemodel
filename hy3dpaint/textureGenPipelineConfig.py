"""
Hunyuan3D-Paint Configuration
Configuration class for the texture generation pipeline
"""
import os


class Hunyuan3DPaintConfig:
    """
    Configuration class for Hunyuan3D-Paint pipeline.
    """
    
    def __init__(self, max_num_view: int = 6, resolution: int = 512):
        """
        Initialize the configuration.
        
        Args:
            max_num_view (int): Maximum number of views (6-9)
            resolution (int): Resolution for generated textures (512 or 768)
        """
        self.max_num_view = max_num_view
        self.resolution = resolution
        self.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        
    def validate(self):
        """
        Validate configuration parameters.
        """
        if not 6 <= self.max_num_view <= 9:
            raise ValueError("max_num_view must be between 6 and 9")
            
        if self.resolution not in [512, 768]:
            raise ValueError("resolution must be either 512 or 768")