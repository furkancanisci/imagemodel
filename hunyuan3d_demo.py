"""
Demo Script for Hunyuan3D-2.1
Exact replica of the demo.py from the official repository
"""
import sys
import os

# Add paths for hy3dshape and hy3dpaint modules
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

def main():
    # Shape generation
    print("=== Hunyuan3D-2.1 Shape Generation ===")
    model_path = 'tencent/Hunyuan3D-2.1'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
    
    # Load and process image
    image_path = 'new_test_image.jpg'  # Using our new test image
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
        
    image = Image.open(image_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    
    # Generate mesh
    mesh = pipeline_shapegen(image=image)[0]
    # Use OBJ export to avoid NumPy compatibility issues
    mesh.export('new_model.obj')
    print("Shape generation completed. Mesh saved as 'new_model.obj'")
    
    # Texture generation
    print("\n=== Hunyuan3D-2.1 Texture Generation ===")
    max_num_view = 6  # can be 6 to 9
    resolution = 512  # can be 768 or 512
    conf = Hunyuan3DPaintConfig(max_num_view, resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    paint_pipeline = Hunyuan3DPaintPipeline(conf)
    
    # Generate textured mesh
    output_mesh_path = 'new_model_textured.obj'
    output_mesh_path = paint_pipeline(
        mesh_path="new_model.obj",
        image_path=image_path,
        output_mesh_path=output_mesh_path
    )
    
    print("Texture generation completed.")
    print(f"Textured mesh saved as '{output_mesh_path}'")

if __name__ == "__main__":
    main()