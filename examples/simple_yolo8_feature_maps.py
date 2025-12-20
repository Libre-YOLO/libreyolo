import os
from pathlib import Path
from libreyolo import LIBREYOLO8

def main():
    # Get project root (parent of examples directory)
    project_root = Path(__file__).parent.parent
    
    # Paths relative to project root
    weights = project_root / "weights" / "libreyolo8n.pt"
    image = project_root / "media" / "test_image_1_creative_commons.jpg"
    
    # Initialize model
    model = LIBREYOLO8(model_path=str(weights), size="n", save_feature_maps=True)
    
    # Run inference
    results = model.predict(str(image), save=True)
    
    print(f"Found {results['num_detections']} detections.")
    print(f"Result saved to: {results.get('saved_path')}")

if __name__ == "__main__":
    main()


