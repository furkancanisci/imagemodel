"""
Image to 3D Model Pipeline
"""
import os
import torch
import numpy as np
from PIL import Image
import trimesh
import cv2

# Import our models
from my3d.models.feature_extractor import FeatureExtractor, DepthEstimator, NormalEstimator
from my3d.models.mesh_generator import MeshGenerator, volumetric_from_depth


class ImageTo3D:
    """
    Main class for converting images to 3D models.
    
    Usage:
        from my3d.pipeline import ImageTo3D
        model = ImageTo3D('image.jpg')
        model.export('model.glb')
    """
    
    def __init__(self, image_path, model_type='resnet50'):
        """
        Initialize the ImageTo3D pipeline.
        
        Args:
            image_path (str): Path to the input image file
            model_type (str): Type of feature extractor to use ('resnet50', 'vit')
        """
        self.image_path = image_path
        self.model_type = model_type
        self.image = None
        self.mesh = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_image()
        self._initialize_models()
        
    def _load_image(self):
        """Load and preprocess the input image."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
            
        self.image = Image.open(self.image_path).convert('RGB')
        print(f"Loaded image: {self.image_path} with size {self.image.size}")
        
    def _initialize_models(self):
        """Initialize the deep learning models."""
        print("Initializing models...")
        self.feature_extractor = FeatureExtractor(model_type=self.model_type).to(self.device)
        self.depth_estimator = DepthEstimator().to(self.device)
        self.normal_estimator = NormalEstimator().to(self.device)
        self.mesh_generator = MeshGenerator(method='marching_cubes')
        print("Models initialized.")
        
    def _preprocess_image(self):
        """
        Preprocess the image for model input.
        
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert PIL image to numpy array
        img_array = np.array(self.image)
        
        # Resize to model input size (224x224 for ResNet)
        img_resized = cv2.resize(img_array, (224, 224))
        
        # Convert BGR to RGB if needed
        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_rgb = img_resized
        else:
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
        # Normalize image
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
        
    def generate_mesh(self):
        """
        Generate 3D mesh from the input image using deep learning models.
        """
        print("Generating 3D mesh from image...")
        
        # Preprocess image
        img_tensor = self._preprocess_image()
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            print(f"Extracted features with shape: {features.shape}")
            
            # Estimate depth and normal maps
            depth_map = self.depth_estimator(features)
            normal_map = self.normal_estimator(features)
            
            print(f"Depth map shape: {depth_map.shape}")
            print(f"Normal map shape: {normal_map.shape}")
            
            # Convert to numpy arrays
            depth_np = depth_map.squeeze().cpu().numpy()
            normal_np = normal_map.squeeze().cpu().numpy()
            
            # Convert to volumetric representation
            volume = volumetric_from_depth(depth_np, normal_np)
            print(f"Volume shape: {volume.shape}")
            
            # Generate mesh
            self.mesh = self.mesh_generator.generate_mesh(volume)
            print("Mesh generation completed.")
        
    def export(self, output_path):
        """
        Export the generated 3D mesh to a file.
        
        Args:
            output_path (str): Path where the 3D model will be saved
        """
        if self.mesh is None:
            self.generate_mesh()
            
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Export the mesh
        # Handle different export formats
        file_ext = os.path.splitext(output_path)[1].lower()
        
        try:
            if file_ext == '.obj':
                # OBJ export is more stable
                self.mesh.export(output_path, file_type='obj')
                print(f"3D model exported to: {output_path}")
            elif file_ext == '.glb':
                # Try GLB export with error handling
                try:
                    self.mesh.export(output_path, file_type='glb')
                    print(f"3D model exported to: {output_path}")
                except Exception as e:
                    print(f"Warning: Could not export to GLB format: {e}")
                    print("Exporting as OBJ instead...")
                    obj_path = os.path.splitext(output_path)[0] + '.obj'
                    self.mesh.export(obj_path, file_type='obj')
                    print(f"Model exported as OBJ: {obj_path}")
            else:
                # For other formats, try with error handling
                self.mesh.export(output_path)
                print(f"3D model exported to: {output_path}")
        except Exception as e:
            print(f"Error during export: {e}")
            # Fallback to OBJ export
            obj_path = os.path.splitext(output_path)[0] + '.obj' if '.' in output_path else output_path + '.obj'
            self.mesh.export(obj_path, file_type='obj')
            print(f"Fallback: Model exported as OBJ: {obj_path}")