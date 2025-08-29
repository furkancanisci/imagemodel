"""
Example script demonstrating how to use the PracticalImageTo3D pipeline
"""
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from my3d.practical_pipeline import PracticalImageTo3D

def main():
    # Check if an image path was provided
    if len(sys.argv) < 2:
        print("Usage: python practical_example.py <image_path>")
        print("Example: python practical_example.py image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return
    
    try:
        # Create the PracticalImageTo3D pipeline
        print(f"Loading image: {image_path}")
        model = PracticalImageTo3D()
        
        # Generate the 3D model
        model.generate_mesh(image_path)
        
        # Export to different formats
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Export as OBJ (most stable format)
        obj_path = f"{base_name}_practical.obj"
        model.export(obj_path)
        print(f"Model exported as OBJ: {obj_path}")
        
        # Try to export as GLB
        glb_path = f"{base_name}_practical.glb"
        try:
            model.export(glb_path)
            print(f"Model exported as GLB: {glb_path}")
        except Exception as e:
            print(f"Could not export as GLB: {e}")
        
        print("3D model generation completed successfully!")
        
    except Exception as e:
        print(f"Error during 3D model generation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()