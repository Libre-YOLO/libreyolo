import os
from libreyolo import LIBREYOLO

def run_yolo8_inference():
    weights_path = "weights/libreyolo8n.pt"
    test_image = "media/test_image_1_creative_commons.jpg"
    
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Running Inference with LibreYOLO8n ---")
    try:
        model = LIBREYOLO(weights_path, size="n")
        results = model(image=test_image, save=True)
        print(f"Detected {results['num_detections']} objects.")
        if 'saved_path' in results:
            print(f"Result saved to: {results['saved_path']}")
    except Exception as e:
        print(f"Error during YOLOv8 inference: {e}")

if __name__ == "__main__":
    run_yolo8_inference()

