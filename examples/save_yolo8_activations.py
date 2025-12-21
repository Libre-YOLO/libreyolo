import os
from libreyolo import LIBREYOLO

def save_yolo8_activations():
    weights_path = "weights/libreyolo8n.pt"
    test_image = "media/test_image_1_creative_commons.jpg"
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Saving Specific Activations for LibreYOLO8n ---")
    try:
        # Capture specific backbone and neck layers
        layers_to_save = ["backbone_p3", "backbone_p4", "backbone_p5"]
        model = LIBREYOLO(weights_path, size="n", save_feature_maps=layers_to_save)
        
        # Run inference to trigger saving
        model(image=test_image, save=False)
        print(f"Feature maps for {layers_to_save} saved to: runs/feature_maps/")
    except Exception as e:
        print(f"Error saving YOLOv8 activations: {e}")

if __name__ == "__main__":
    save_yolo8_activations()

