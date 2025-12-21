import os
from libreyolo import LIBREYOLO

def export_yolo11():
    weights_path = "weights/libreyolo11n.pt"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        return

    print(f"Exporting {weights_path} to ONNX...")
    try:
        model = LIBREYOLO(weights_path, size="n")
        onnx_path = model.export("libreyolo11n.onnx")
        print(f"Successfully exported to: {onnx_path}")
    except Exception as e:
        print(f"Error exporting YOLOv11: {e}")

if __name__ == "__main__":
    export_yolo11()

