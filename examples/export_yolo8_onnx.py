import os
from libreyolo import LIBREYOLO

def export_yolo8():
    weights_path = "/Users/xuban.ceccon/Documents/GitHub/libreyolo/weights/libreyolo8n.pt"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return

    print(f"Exporting {weights_path} to ONNX...")
    try:
        model = LIBREYOLO(weights_path, size="n")
        onnx_path = model.export("libreyolo8n.onnx")
        print(f"Successfully exported to: {onnx_path}")
    except Exception as e:
        print(f"Error exporting YOLOv8: {e}")

if __name__ == "__main__":
    export_yolo8()

