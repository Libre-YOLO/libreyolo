"""
Weight conversion utility for converting YOLOv8 weights to Libre YOLO format.

IMPORTANT LICENSE NOTICE:
=========================
Any weights obtained from external sources inherit the license terms from where those
weights originated. The converted weights maintain the same license restrictions and
terms as the original source weights. Users must comply with the original license terms
of the source weights they are converting.
"""

import torch
import argparse

# LICENSE NOTICE: Weights inherit the license from their source
print("=" * 70)
print("LICENSE NOTICE: Source weights inherit the license from where")
print("they originated. Converted weights maintain the same license restrictions.")
print("=" * 70)


def convert_weights(source_path, output_path):
    """
    Convert YOLOv8 weights to Libre YOLO format.
    
    LICENSE NOTICE: Any weights from external sources inherit the license from where
    those weights originated. The converted weights maintain the same license
    restrictions as the original source weights.
    """
    print(f"Loading weights from: {source_path}")
    
    try:
        source_data = torch.load(source_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"Error: Could not find {source_path}.")
        return
    except Exception as e:
        error_str = str(e)
        if "module" in error_str.lower() or "import" in error_str.lower() or "unpicklingerror" in error_str.lower():
            print("Error: Missing required dependencies to load this checkpoint.")
            print("Install with: pip install ultralytics")
            return
        else:
            print(f"Error loading checkpoint: {error_str}")
            return
    
    # Extract state_dict
    if 'model' in source_data:
        source_state = source_data['model'].state_dict()
    else:
        source_state = source_data
    
    print(f"Source contains {len(source_state)} keys.")
    
    # Detect prefix
    sample_keys = list(source_state.keys())
    prefix = ""
    if any(k.startswith("model.model.") for k in sample_keys):
        prefix = "model.model."
    elif any(k.startswith("model.") for k in sample_keys):
        prefix = "model."
    
    print(f"Detected key prefix: '{prefix}'")
    
    # Block mapping
    block_map = {
        "0.": "backbone.p1.",
        "1.": "backbone.p2.",
        "2.": "backbone.c2f1.",
        "3.": "backbone.p3.",
        "4.": "backbone.c2f2.",
        "5.": "backbone.p4.",
        "6.": "backbone.c2f3.",
        "7.": "backbone.p5.",
        "8.": "backbone.c2f4.",
        "9.": "backbone.sppf.",
        "12.": "neck.c2f21.",
        "15.": "neck.c2f11.",
        "16.": "neck.conv1.",
        "18.": "neck.c2f12.",
        "19.": "neck.conv2.",
        "21.": "neck.c2f22.",
    }
    
    print("Converting weights...")
    new_state_dict = {}
    
    for key, value in source_state.items():
        # Remove prefix
        if prefix and key.startswith(prefix):
            rel_key = key[len(prefix):]
        else:
            rel_key = key
        
        new_key = ""
        mapped = False
        
        # Handle Head (Layer 22)
        if rel_key.startswith("22."):
            if ".dfl." in rel_key:
                new_key = "dfl.conv.weight"
                new_state_dict[new_key] = value
                continue
            
            parts = rel_key.split('.')
            if len(parts) < 4:
                continue
            
            branch = parts[1]  # cv2 (Box) or cv3 (Class)
            stride_idx = parts[2]  # 0 (8s), 1 (16s), 2 (32s)
            layer_idx = parts[3]  # 0, 1, or 2
            
            # Map stride
            if stride_idx == '0':
                head_name = "head8"
            elif stride_idx == '1':
                head_name = "head16"
            elif stride_idx == '2':
                head_name = "head32"
            else:
                continue
            
            # Map branch
            if branch == 'cv2':  # Box
                if layer_idx == '0':
                    module = "conv11"
                elif layer_idx == '1':
                    module = "conv12"
                elif layer_idx == '2':
                    module = "cnn1"
                else:
                    continue
            elif branch == 'cv3':  # Class
                if layer_idx == '0':
                    module = "conv21"
                elif layer_idx == '1':
                    module = "conv22"
                elif layer_idx == '2':
                    module = "cnn2"
                else:
                    continue
            else:
                continue
            
            suffix = ".".join(parts[4:])
            new_key = f"{head_name}.{module}.{suffix}"
            mapped = True
        
        # Handle Backbone & Neck
        else:
            for src_idx, tgt_prefix in block_map.items():
                if rel_key.startswith(src_idx):
                    new_key = tgt_prefix + rel_key[len(src_idx):]
                    mapped = True
                    break
        
        if not mapped:
            continue
        
        # Terminology updates
        new_key = new_key.replace("cv1", "conv1")
        new_key = new_key.replace("cv2", "conv2")
        new_key = new_key.replace(".m.", ".bottlenecks.")
        new_key = new_key.replace(".bn.", ".batchnorm.")
        new_key = new_key.replace(".conv.", ".cnn.")
        
        new_state_dict[new_key] = value
    
    # LICENSE NOTICE: Converted weights inherit license from source
    print("LICENSE REMINDER: Converted weights inherit license from source weights.")
    torch.save(new_state_dict, output_path)
    print(f"Conversion complete! Saved {len(new_state_dict)} keys to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv8 weights to Libre YOLO format")
    parser.add_argument("--source", type=str, required=True, help="Path to source YOLOv8 weights file")
    parser.add_argument("--output", type=str, required=True, help="Path to output converted weights file")
    
    args = parser.parse_args()
    convert_weights(source_path=args.source, output_path=args.output)
