import os
from libreyolo import LIBREYOLO

def save_yolo11_activations():
    weights_path = "weights/libreyolo11n.pt"
    test_image = "media/test_image_1_creative_commons.jpg"
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Saving Activations for LibreYOLO11n (All Layers) ---")
    try:
        # Set save_feature_maps=True to capture all layers
        model = LIBREYOLO(weights_path, size="n", save_feature_maps=True)
        
        # Run inference to trigger saving
        model(image=test_image, save=False)
        print(f"Feature maps for all layers saved to: runs/feature_maps/")
    except Exception as e:
        print(f"Error saving YOLOv11 activations: {e}")

if __name__ == "__main__":
    save_yolo11_activations()

