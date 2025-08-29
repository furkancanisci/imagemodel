# Image to 3D Model Generator

This project generates high-quality 3D meshes from single or multiple images using deep learning techniques.

## Features

- Converts 1 or more images (.jpg/.png) to 3D mesh
- Outputs clean topology 3D meshes in OBJ, GLB, or FBX formats
- Prioritizes geometry generation (texture support coming in later phases)

## Pipeline

1. Feature extraction using Vision Transformer or ResNet-based encoder
2. Depth map and normal estimation
3. Triplane or volumetric representation
4. Mesh generation using Marching Cubes or Poisson reconstruction

## Requirements

- Python 3.10+
- PyTorch 2.5.1+
- CUDA-compatible GPU (recommended)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd imagemodel
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### As a Python Module

```python
from my3d.pipeline import ImageTo3D

# Create the pipeline
model = ImageTo3D('image.jpg')

# Generate and export the 3D model
model.export('model.glb')
```

### Command Line

```bash
python example.py image.jpg
```

This will generate both `image.obj` and `image.glb` files.

## Project Structure

```
imagemodel/
├── my3d/
│   ├── __init__.py
│   ├── pipeline.py      # Main ImageTo3D class
│   ├── models/          # Neural network models
│   │   ├── __init__.py
│   │   ├── feature_extractor.py  # Feature extraction, depth & normal estimation
│   │   └── mesh_generator.py     # Mesh generation algorithms
│   └── utils/           # Utility functions
│       ├── __init__.py
│       └── visualization.py      # Visualization utilities
├── example.py           # Example usage script
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Implementation Details

Our current implementation includes:

1. **Feature Extraction**: Using ResNet-50 or Vision Transformer to extract image features
2. **Depth & Normal Estimation**: Neural networks that estimate depth and surface normals from image features
3. **Volumetric Representation**: Converting depth maps to 3D volumetric data
4. **Mesh Generation**: Using Marching Cubes algorithm to generate 3D meshes from volumetric data

## Future Improvements

To create a production-ready system like Hunyuan3D-2.1, the following enhancements would be needed:

1. **Advanced Models**: 
   - Implement Triplane or NeRF-based representations
   - Use diffusion models for higher quality generation
   - Add Physically-Based Rendering (PBR) texture synthesis

2. **Improved Pipeline**:
   - Multi-view image processing (2-6 images)
   - Better depth and normal estimation networks
   - Advanced mesh generation with Poisson reconstruction
   - Texture mapping capabilities

3. **Performance Optimizations**:
   - Optimize for lower VRAM usage
   - Add support for low VRAM mode
   - Implement model quantization techniques

4. **User Interface**:
   - Add Gradio or Streamlit web interface
   - Create a command-line interface with more options
   - Add batch processing capabilities

## References

- Based on concepts from Hunyuan3D-2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- PyTorch 3D: https://pytorch3d.org/
- Trimesh: https://trimsh.org/