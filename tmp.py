
import json
import os
data_path='../autodl-tmp/ROP_shen'
with open(os.path.join(data_path,'annotations.json')) as f:
    data_dict=json.load(f)

with open('./label_v1.json') as f:
    label_v1=json.load(f)
with open(os.path.join(data_path,'split','all.json')) as f:
    split_all=json.load(f)

pred=[]
labels=[]
for image_name in split_all['test']:
    if data_dict[image_name]['stage']>0:
        labels.append(1)
    else:
        labels.append(0)
    if 'ridge_seg_path' in data_dict[image_name]['ridge_seg']:
        pred.append(1)
    else:
        pred.append(0)
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

# Convert lists to numpy arrays for efficient logical operations
import numpy as np
labels_np = np.array(labels)
pred_np = np.array(pred)

# Calculate accuracy
accuracy = accuracy_score(labels_np, pred_np)
print(f"Accuracy: {accuracy:.4f}")

# Calculate AUC
# Note: AUC requires probability scores, but if you only have binary predictions, you might consider a different metric or ensure your predictions can be interpreted as probabilities.
# Here, we proceed by considering binary predictions for demonstration.
# Ensure your labels and predictions are properly formatted for AUC calculation, or use a more appropriate metric for binary predictions.
# try:
#     auc = roc_auc_score(labels_np, pred_np)
#     print(f"AUC: {auc:.4f}")
# except ValueError as e:
#     print(e)

# Calculate recall for positive predictions
# For binary classification, "positive" means the class of interest is labeled '1'
# Recall in this context is the proportion of actual positives that were identified correctly
recall = recall_score(labels_np, pred_np, pos_label=1)
print(f"Recall (Positive Predictions): {recall:.4f}")

# Additional metric: Recall for cases where both prediction and label are positive
# Note: This calculation is conceptual. In a binary classification, recall is usually calculated as TP / (TP + FN).
# Here, we illustrate a conceptual calculation based on the given condition.
positive_cases = (labels_np == 1) & (pred_np == 1)
if np.any(labels_np == 1):
    recall_positive = np.sum(positive_cases) / np.sum(labels_np == 1)
    print(f"Recall (pred > 0 & label > 0 / label > 0): {recall_positive:.4f}")
else:
    print("No positive labels in dataset.")
