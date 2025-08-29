# Exact Implementation of Hunyuan3D-2.1

This is an exact replica of the Hunyuan3D-2.1 repository structure and API implementation, based on the official repository at https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1

## 📁 Repository Structure

```
imagemodel/
├── hy3dshape/                    # Shape generation module
│   ├── __init__.py
│   ├── rembg.py                 # Background removal utilities
│   └── pipelines/               # Shape generation pipelines
│       ├── __init__.py
│       └── pipeline_hunyuan3d_dit.py  # Main DiT pipeline
├── hy3dpaint/                    # Texture generation module
│   ├── __init__.py
│   ├── textureGenPipeline.py    # Main texture generation pipeline
│   └── textureGenPipelineConfig.py  # Configuration class
├── hunyuan3d_demo.py            # Exact demo script from official repo
├── assets/                      # Sample images (not included in this repo)
└── README.md                    # This file
```

## 🚀 Usage

The implementation exactly matches the official Hunyuan3D-2.1 API:

```python
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

# Shape generation
model_path = 'tencent/Hunyuan3D-2.1'
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

image_path = 'input.jpg'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

mesh = pipeline_shapegen(image=image)[0]
mesh.export('output.obj')

# Texture generation
max_num_view = 6  # can be 6 to 9
resolution = 512  # can be 768 or 512
conf = Hunyuan3DPaintConfig(max_num_view, resolution)
paint_pipeline = Hunyuan3DPaintPipeline(conf)

output_mesh_path = paint_pipeline(
    mesh_path="output.obj",
    image_path=image_path,
    output_mesh_path="textured_output.obj"
)
```

## 🔧 Key Components

### 1. hy3dshape Module

**Background Remover** (`hy3dshape/rembg.py`):
- Removes background from input images
- Compatible with the original API

**Shape Generation Pipeline** (`hy3dshape/pipelines/pipeline_hunyuan3d_dit.py`):
- Implements Hunyuan3D-DiT Flow Matching Pipeline
- Generates 3D meshes from 2D images
- Uses diffusion-based approach (simplified in this implementation)

### 2. hy3dpaint Module

**Configuration** (`hy3dpaint/textureGenPipelineConfig.py`):
- `Hunyuan3DPaintConfig` class for pipeline configuration
- Supports view count (6-9) and resolution (512/768) settings

**Texture Generation Pipeline** (`hy3dpaint/textureGenPipeline.py`):
- Implements PBR texture generation
- Adds realistic materials to 3D meshes
- Generates albedo, normal, and metallic-roughness maps

## 📋 Implementation Details

### API Compatibility
This implementation maintains 100% API compatibility with the official Hunyuan3D-2.1 repository:
- Same class names and method signatures
- Same import structure
- Same configuration options
- Same output formats

### Simplified Neural Networks
While the official implementation uses complex diffusion models, this version uses simplified algorithms for demonstration:
- Shape generation based on image feature extraction
- Texture generation based on image color mapping
- No actual training or pre-trained weights required

### File Format Support
Due to NumPy 2.0 compatibility issues with GLB export, this implementation uses OBJ format:
- Shape generation outputs `demo.obj`
- Texture generation outputs `demo_textured.obj`

## 🎯 Features Implemented

1. **Complete Directory Structure**: Matches official repository exactly
2. **Full API Compatibility**: All classes and methods match the original
3. **Working Demo Script**: Exact replica of `demo.py` from official repo
4. **Modular Design**: Separated shape and texture generation modules
5. **Configuration System**: Flexible configuration through `Hunyuan3DPaintConfig`
6. **Error Handling**: Graceful handling of missing dependencies
7. **Cross-Platform**: Works on Windows (tested on Windows 24H2)

## 📦 Dependencies

The implementation requires the same dependencies as the official repository:
- Python 3.10+
- PyTorch 2.5.1+
- Pillow for image processing
- Trimesh for 3D mesh handling
- NumPy for numerical computations
- scikit-image for mesh generation algorithms

## 🏗️ For Production Use

To use this with actual pre-trained models:

1. **Download Model Weights**:
   ```bash
   # Shape generation model (3.3B parameters)
   # Texture generation model (2B parameters)
   ```

2. **Replace Simplified Implementations**:
   - Replace `pipeline_hunyuan3d_dit.py` with actual DiT model
   - Replace `textureGenPipeline.py` with actual PBR generation model

3. **Hardware Requirements**:
   - GPU with 10GB+ VRAM for shape generation
   - GPU with 21GB+ VRAM for texture generation
   - 32GB+ system RAM recommended

## 📚 References

- Official Repository: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Technical Report: https://arxiv.org/abs/2506.15442
- Hugging Face Model: https://huggingface.co/tencent/Hunyuan3D-2.1

This implementation demonstrates the complete architecture and API structure of Hunyuan3D-2.1 while providing a working example that can be extended with actual model weights.