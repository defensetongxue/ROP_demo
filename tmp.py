import torch
import numpy as np
from config import get_config
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import os
import json
from PIL import Image

# Ensure required directories are created
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("experiments", exist_ok=True)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load configuration
args = get_config()

# Read annotations and test splits
with open(os.path.join(args.data_path, 'annotations.json'), 'r') as f:
    data_dict = json.load(f)
with open(os.path.join(args.data_path, 'split', 'all.json'), 'r') as f:
    split_all = json.load(f)['test']

# Load model predictions
with open('./model_predit.json', 'r') as f:
    pred_result = json.load(f)

# Initialize lists for metrics calculation
all_targets = []
pred_list = []

# Process each image in the test split
for image_name in split_all:
    data = data_dict[image_name]
    label = int(data['stage'] > 0)  # Binary label based on stage presence
    pred = pred_result[image_name]  # Prediction from model

    # Append actual and predicted values to lists
    all_targets.append(label)
    pred_list.append(pred)

# Calculate metrics
accuracy = accuracy_score(all_targets, pred_list)
auc = roc_auc_score(all_targets, pred_list)
recall = recall_score(all_targets, pred_list, pos_label=1)  # Assuming positive label is '1'

# Print the calculated metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")

# Optionally, save these metrics to a file or handle them as needed
metrics_path = "./experiments/metrics.json"
with open(metrics_path, 'w') as f:
    json.dump({'accuracy': accuracy, 'auc': auc, 'recall': recall}, f, indent=4)

print("Metrics saved to", metrics_path)

