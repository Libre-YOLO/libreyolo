"""
Weight conversion utility for converting YOLOv11 weights to Libre YOLO format.

IMPORTANT LICENSE NOTICE:
=========================
Any weights obtained from external sources inherit the license terms from where those
weights originated. The converted weights maintain the same license restrictions and
terms as the original source weights. Users must comply with the original license terms
of the source weights they are converting.
"""

import torch
import argparse
from ultralytics import YOLO

# LICENSE NOTICE: Weights inherit the license from their source
print("=" * 70)
print("LICENSE NOTICE: Source weights inherit the license from where")
print("they originated. Converted weights maintain the same license restrictions.")
print("=" * 70)


def convert_weights(source_path, output_path):
    """
    Convert YOLOv11 weights to Libre YOLO format.
    
    LICENSE NOTICE: Any weights from external sources inherit the license from where
    those weights originated. The converted weights maintain the same license
    restrictions as the original source weights.
    """
    print(f"Loading weights from: {source_path}")
    
    try:
        # Load using Ultralytics to get the model structure
        modelo = YOLO(source_path)
        source_state = modelo.model.state_dict()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"Source contains {len(source_state)} keys.")
    
    # Block mapping for YOLO11 backbone and neck
    # Based on actual YOLO11 layer indices
    block_map = {
        "0.": "backbone.p1.",      # Conv
        "1.": "backbone.p2.",      # Conv
        "2.": "backbone.c2f1.",    # C3k2
        "3.": "backbone.p3.",      # Conv
        "4.": "backbone.c2f2.",    # C3k2
        "5.": "backbone.p4.",      # Conv
        "6.": "backbone.c2f3.",    # C3k2 (with C3k)
        "7.": "backbone.p5.",      # Conv
        "8.": "backbone.c2f4.",    # C3k2 (with C3k)
        "9.": "backbone.sppf.",    # SPPF
        "10.": "backbone.c2psa.",  # C2PSA
        "13.": "neck.c2f21.",      # C3k2 (upsample path)
        "16.": "neck.c2f11.",      # C3k2 (upsample path)
        "17.": "neck.conv1.",      # Conv (downsample)
        "19.": "neck.c2f12.",      # C3k2 (downsample path)
        "20.": "neck.conv2.",      # Conv (downsample)
        "22.": "neck.c2f22.",      # C3k2 (downsample path, with C3k)
    }
    
    print("Converting weights...")
    new_state_dict = {}
    
    for key, value in source_state.items():
        # Remove "model." prefix
        if key.startswith("model."):
            rel_key = key[6:]  # Remove "model."
        else:
            rel_key = key
        
        new_key = None
        mapped = False
        
        # Skip num_batches_tracked
        if "num_batches_tracked" in rel_key:
            continue
        
        # Handle DFL (Layer 23)
        if rel_key.startswith("23.dfl."):
            new_key = "dfl." + rel_key[7:]  # Remove "23.dfl."
            # DFL uses .conv, not .cnn
            if ".conv" in new_key:
                new_key = new_key.replace(".conv", ".conv")  # Keep as conv
            mapped = True
        
        # Handle Head (Layer 23)
        elif rel_key.startswith("23."):
            parts = rel_key.split('.')
            if len(parts) < 3:
                continue
            
            branch = parts[1]  # cv2 (Box) or cv3 (Class)
            stride_idx = parts[2]  # 0 (head8), 1 (head16), 2 (head32)
            
            # Map stride to head name
            if stride_idx == '0':
                head_name = "head8"
            elif stride_idx == '1':
                head_name = "head16"
            elif stride_idx == '2':
                head_name = "head32"
            else:
                continue
            
            # Handle Box Branch (cv2)
            if branch == 'cv2':
                if len(parts) < 4:
                    continue
                layer_idx = parts[3]  # 0, 1, or 2
                
                if layer_idx == '0':
                    module = "conv11"
                    # cv2.X.0.conv.weight -> conv11.cnn.weight
                    # cv2.X.0.bn.weight -> conv11.batchnorm.weight
                    suffix = '.'.join(parts[4:])  # Everything after cv2.X.0
                elif layer_idx == '1':
                    module = "conv12"
                    suffix = '.'.join(parts[4:])  # Everything after cv2.X.1
                elif layer_idx == '2':
                    module = "cnn1"
                    # cv2.X.2.weight -> cnn1.weight (no .conv, no .bn)
                    suffix = '.'.join(parts[4:])  # Everything after cv2.X.2
                else:
                    continue
                
                new_key = f"{head_name}.{module}.{suffix}" if suffix else f"{head_name}.{module}"
                mapped = True
            
            # Handle Class Branch (cv3)
            elif branch == 'cv3':
                if len(parts) < 4:
                    continue
                
                # cv3 has nested structure: cv3.X.Y.Z.*
                # Y=0: cv3.X.0.0.* -> conv21.0.*, cv3.X.0.1.* -> conv21.1.*
                # Y=1: cv3.X.1.0.* -> conv22.0.*, cv3.X.1.1.* -> conv22.1.*
                # Y=2: cv3.X.2.* -> cnn2.*
                y_idx = parts[3]  # 0, 1, or 2
                
                if y_idx == '0':
                    # cv3.X.0.Z.* -> conv21.Z.*
                    if len(parts) >= 5:
                        z_idx = parts[4]  # 0 or 1
                        module = "conv21"
                        suffix = f"{z_idx}." + '.'.join(parts[5:]) if len(parts) > 5 else z_idx
                        new_key = f"{head_name}.{module}.{suffix}" if suffix else f"{head_name}.{module}"
                        mapped = True
                elif y_idx == '1':
                    # cv3.X.1.Z.* -> conv22.Z.*
                    if len(parts) >= 5:
                        z_idx = parts[4]  # 0 or 1
                        module = "conv22"
                        suffix = f"{z_idx}." + '.'.join(parts[5:]) if len(parts) > 5 else z_idx
                        new_key = f"{head_name}.{module}.{suffix}" if suffix else f"{head_name}.{module}"
                        mapped = True
                elif y_idx == '2':
                    # cv3.X.2.* -> cnn2 (final, no batch norm)
                    module = "cnn2"
                    suffix = '.'.join(parts[4:])  # Everything after cv3.X.2
                    new_key = f"{head_name}.{module}.{suffix}" if suffix else f"{head_name}.{module}"
                    mapped = True
        
        # Handle Backbone & Neck
        if not mapped:
            for src_idx, tgt_prefix in block_map.items():
                if rel_key.startswith(src_idx):
                    new_key = tgt_prefix + rel_key[len(src_idx):]
                    mapped = True
                    break
        
        if not mapped:
            # Skip layer 22 (shared head processing) - not in LibreYOLO structure
            if rel_key.startswith("22."):
                continue
            print(f"Warning: Unmapped key: {rel_key}")
            continue
        
        # Terminology updates (order matters!)
        # First replace cv1/cv2/cv3 with conv1/conv2/conv3
        new_key = new_key.replace("cv1", "conv1")
        new_key = new_key.replace("cv2", "conv2")
        new_key = new_key.replace("cv3", "conv3")
        
        # Then replace .m. with .bottlenecks.
        new_key = new_key.replace(".m.", ".bottlenecks.")
        
        # Replace .bn. with .batchnorm. (but not if it's part of a word)
        new_key = new_key.replace(".bn.", ".batchnorm.")
        
        # Replace .conv. with .cnn. (but only when surrounded by dots)
        # EXCEPT for DFL which uses .conv
        # This is safe because we've already replaced cv1/cv2/cv3
        if not new_key.startswith("dfl."):
            new_key = new_key.replace(".conv.", ".cnn.")
        
        new_state_dict[new_key] = value
    
    # LICENSE NOTICE: Converted weights inherit license from source
    print("LICENSE REMINDER: Converted weights inherit license from source weights.")
    print(f"Converted {len(new_state_dict)} keys.")
    
    # Check for missing expected keys
    expected_prefixes = ["backbone.", "neck.", "head", "dfl."]
    found_prefixes = set()
    for key in new_state_dict.keys():
        for prefix in expected_prefixes:
            if key.startswith(prefix):
                found_prefixes.add(prefix)
                break
    
    print(f"Found keys for: {sorted(found_prefixes)}")
    
    torch.save(new_state_dict, output_path)
    print(f"Conversion complete! Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLOv11 weights to Libre YOLO format")
    parser.add_argument("--source", type=str, required=True, help="Path to source YOLOv11 weights file")
    parser.add_argument("--output", type=str, required=True, help="Path to output converted weights file")
    
    args = parser.parse_args()
    convert_weights(source_path=args.source, output_path=args.output)
