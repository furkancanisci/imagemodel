"""
Feature extraction using Vision Transformer or ResNet-based encoder
"""
import torch
import torch.nn as nn
from torchvision import models
import numpy as np


class FeatureExtractor(nn.Module):
    """
    Feature extractor using Vision Transformer or ResNet-based encoder
    """
    
    def __init__(self, model_type='resnet50', pretrained=True):
        """
        Initialize the feature extractor.
        
        Args:
            model_type (str): Type of model to use ('resnet50', 'vit')
            pretrained (bool): Whether to use pretrained weights
        """
        super(FeatureExtractor, self).__init__()
        self.model_type = model_type
        
        if model_type == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_type == 'vit':
            # For Vision Transformer, we would use a transformer-based model
            # This is a placeholder implementation
            self.model = models.vit_b_16(pretrained=pretrained)
            # Remove the final classification head
            self.model.heads = nn.Identity()
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def forward(self, x):
        """
        Extract features from input image.
        
        Args:
            x (torch.Tensor): Input image tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        features = self.model(x)
        return features


class DepthEstimator(nn.Module):
    """
    Depth map estimation from image features
    """
    
    def __init__(self, input_dim=2048, output_size=(32, 32)):
        """
        Initialize the depth estimator.
        
        Args:
            input_dim (int): Dimension of input features
            output_size (tuple): Size of output depth map
        """
        super(DepthEstimator, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        
        # Simple decoder to generate depth map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Estimate depth map from features.
        
        Args:
            features (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Estimated depth map
        """
        # Reshape features to 2D if needed
        if len(features.shape) == 2:
            # Assume features are [batch_size, channels]
            # Reshape to [batch_size, channels, 1, 1]
            features = features.view(features.size(0), features.size(1), 1, 1)
            
        # Add more layers to get to desired output size
        # For a 32x32 output, we need more upsampling
        features_upsampled = features
        while features_upsampled.shape[2] < 2:
            features_upsampled = nn.functional.interpolate(
                features_upsampled, 
                scale_factor=2, 
                mode='nearest'
            )
            
        depth_map = self.decoder(features_upsampled)
        return depth_map


class NormalEstimator(nn.Module):
    """
    Normal map estimation from image features
    """
    
    def __init__(self, input_dim=2048, output_size=(32, 32)):
        """
        Initialize the normal estimator.
        
        Args:
            input_dim (int): Dimension of input features
            output_size (tuple): Size of output normal map
        """
        super(NormalEstimator, self).__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        
        # Simple decoder to generate normal map
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Normal vectors should be in [-1, 1] range
        )
        
    def forward(self, features):
        """
        Estimate normal map from features.
        
        Args:
            features (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Estimated normal map
        """
        # Reshape features to 2D if needed
        if len(features.shape) == 2:
            # Assume features are [batch_size, channels]
            # Reshape to [batch_size, channels, 1, 1]
            features = features.view(features.size(0), features.size(1), 1, 1)
            
        # Add more layers to get to desired output size
        # For a 32x32 output, we need more upsampling
        features_upsampled = features
        while features_upsampled.shape[2] < 2:
            features_upsampled = nn.functional.interpolate(
                features_upsampled, 
                scale_factor=2, 
                mode='nearest'
            )
            
        normal_map = self.decoder(features_upsampled)
        return normal_map