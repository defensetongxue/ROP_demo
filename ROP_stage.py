import os
import json
import torch
from PIL import Image
import numpy as np
from config import get_config
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score
from util.stage import visual_sentences, crop_patches
from util.ridge_segment import k_max_values_and_indices
from torchvision import transforms
from ROPStageModel import build_model
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)
args = get_config()
with open(args.stage_cfg,'r') as f:
    args.configs=json.load(f)
# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")
visual_dir=os.path.join(args.result_path,'ROP_stage')
os.makedirs(visual_dir, exist_ok=True)
os.system(f'rm -rf {visual_dir}/*')
for i in ['match','miss']:
    os.makedirs(os.path.join(visual_dir,i),exist_ok=True)
    for j in ['0','1','2','3']:
        os.makedirs(os.path.join(visual_dir,i,j),exist_ok=True)
        
        

# Create the model and criterion
model = build_model(args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.model_dir,'ROP_stage',f"{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.model_dir,'ROP_stage',f"{args.configs['save_name']}")))
model.eval()


all_predictions = []
all_targets = []
probs_list = []
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
    
img_norm=transforms.Compose([
            transforms.Resize((args.configs["resize"],args.configs["resize"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD)])
probs_list=[]
labels_list=[]
pred_list=[]   
with open(os.path.join(args.data_path,'split','clr.json')) as f:
    split_all=json.load(f)['test']
model_prediction_path = os.path.join(visual_dir,'model_prediction.json')
model_prediction={}
visual_patch_size= 200
save_visual_global =False

with torch.no_grad():
    for image_name in split_all:
        data=data_dict[image_name]
        label=int(data['stage'])
        img=Image.open(data["image_path"]).convert("RGB")
        inputs=[]
        if data['ridge_seg']["max_val"]<args.configs['judge_threshold']:
            bc_prob=np.zeros((1,4),dtype=float)
            bc_prob[0,0]=1.
            bc_pred=0
        else:
            if "ridge_seg_path" not in data['ridge_seg']:
                print(data['ridge_seg'])
            output_img=Image.open(data['ridge_seg']["ridge_seg_path"]).convert('L')
            output_img=np.array(output_img)/255
            maxval,pred_point=k_max_values_and_indices(output_img.squeeze(),args.configs["ridge_seg_number"],r=60,threshold=0.3)
            value_list=[]
            point_list=[]
            for value in maxval:
                value=round(float(value),2)
                value_list.append(value)
            for y,x in pred_point:
                point_list.append([int(x),int(y)])
                
                
            sample_visual=[]
            for (x,y),val in zip(point_list,value_list):
                
                sample_visual.append([x,y])
                patch=crop_patches(img,args.configs["patch_size"],x,y)
                patch=img_norm(patch)
                inputs.append(patch.unsqueeze(0))
                if val<args.configs['sample_low_threshold']:
                    break
                
            inputs=torch.cat(inputs,dim=0)

            outputs=model(inputs.to(device))
            probs = torch.softmax(outputs.cpu(), axis=1)
            # output shape is bc,num_class
            # get pred for each patch
            pred_labels = torch.argmax(probs, dim=1)
            # get the max predict  label for this batch ( as  bc_pred)
            bc_pred= int(torch.max(pred_labels))
            # select the patch whose preds_label is equal to bc_pred
            matching_indices = torch.where(pred_labels == bc_pred)[0]
            selected_probs = probs[matching_indices]
            # mean these selectes patches probs as bc_porb
            bc_prob = torch.mean(selected_probs, dim=0)
            bc_prob[-1]=1-torch.sum(bc_prob[:-1])
            bc_prob=bc_prob.unsqueeze(0).numpy()
            bc_prob = np.insert(bc_prob, 0, 0, axis=1)
            
            bc_pred+=1
            # visual the mismatch version
                
            visual=False      
            # if args.visual_miss and label!=bc_pred:
            #     save_path=os.path.join(visual_dir,'miss',str(label),image_name)
            # elif args.visual_match and label==bc_pred:
            #     save_path=os.path.join(visual_dir,'match',str(label),image_name)
            # else:
            #     visual=False # do not visual
            # if visual:
            #     # Get top k firmest predictions for bc_pred class
            #     top_k = min(args.k,matching_indices.shape[0])  # Assuming args.k is defined and valid   
            #     class_probs = probs[:, bc_pred-1]  # Extract probabilities for bc_pred class
            #     top_k_values, top_k_indices = torch.topk(class_probs, k=top_k)
            #     visual_point=[]
            #     visual_confidence=[]
            #     for val,idx in zip(top_k_values,top_k_indices):
            #         x,y=point_list[idx]
            #         visual_point.append([int(x),int(y)])
            #         visual_confidence.append(round(float(val),2))
            #     visual_sentences(
            #         data_dict[image_name]['image_path'],
            #         points=visual_point,
            #         patch_size=visual_patch_size,
            #         text=f"label: {label}",
            #         confidences=visual_confidence,
            #         label=bc_pred,
            #         save_path=save_path,
            #         sample_visual=sample_visual
            #         )
        probs_list.extend(bc_prob)
        labels_list.append(label)
        pred_list.append(bc_pred)
        model_prediction[image_name]=bc_pred
probs_list = np.vstack(probs_list)
pred_labels = np.array(pred_list)
labels_list = np.array(labels_list)

# Calculate metrics for classes 1, 2, and 3
results = {'recall': {}, 'auc': {}, 'f1': {}, 'accuracy': {}}
for class_idx, class_name in zip([1, 2, 3], ["Class 1", "Class 2", "Class 3"]):
    y_true = (labels_list == class_idx).astype(int)
    y_pred = (pred_labels == class_idx).astype(int)

    # Recall
    recall = recall_score(y_true, y_pred)
    results['recall'][class_name] = recall

    # AUC
    auc_score = roc_auc_score(y_true, probs_list[:, class_idx])
    results['auc'][class_name] = auc_score

    # F1 Score
    f1 = f1_score(y_true, y_pred)
    results['f1'][class_name] = f1

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    results['accuracy'][class_name] = accuracy

# Save results to JSON file
os.makedirs('./experiments', exist_ok=True)
with open('./experiments/stage_record.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Metrics calculated and saved successfully.")