import os
from libreyolo import LIBREYOLO11

def main():
    # Paths relative to project root
    weights = "weights/libreyolo11n.pt"
    image = "media/test_image_1_creative_commons.jpg"
    
    # Initialize model
    model = LIBREYOLO11(model_path=weights, size="n")
    
    # Run inference
    results = model.predict(image, save=True)
    
    print(f"Found {results['num_detections']} detections.")
    print(f"Result saved to: {results.get('saved_path')}")

if __name__ == "__main__":
    main()

