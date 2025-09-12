#!/usr/bin/env python3
"""
Inspect the saved model structure to understand the key names and create proper model.
"""

import torch

# Load the saved model to see its structure
model_path = "/Users/lakshmikotaru/Documents/sar_oil_slick_detection/hf_deployment/oil_slick_resnet_subset.pth"
state_dict = torch.load(model_path, map_location='cpu')

print("Keys in saved model state_dict:")
print("="*50)
for key in sorted(state_dict.keys())[:30]:  # Show first 30 keys
    print(f"  {key}")

print("\n...")
print(f"\nTotal keys: {len(state_dict.keys())}")

# Look at the structure
resnet_keys = [k for k in state_dict.keys() if k.startswith('resnet.')]
head_keys = [k for k in state_dict.keys() if not k.startswith('resnet.')]

print(f"\nKeys starting with 'resnet.': {len(resnet_keys)}")
print(f"Keys NOT starting with 'resnet.': {len(head_keys)}")

print("\nNon-resnet keys (likely segmentation head):")
for key in head_keys:
    print(f"  {key}")
