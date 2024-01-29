import json
import os
import torch
from config import get_config
from torchvision import transforms
from util.tools import get_instance
from util.ridge_segment import visual_mask,k_max_values_and_indices
import ridgeSegmentModel as models
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score
import numpy as np
# Parse arguments
import time
torch.manual_seed(0)
np.random.seed(0)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
args = get_config()
with open(args.ridge_seg_cfg,'r') as f:
    args.configs=json.load(f)
# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")
visual_dir=os.path.join(args.result_path,'ridgeSegmentation')
os.makedirs(visual_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)
os.system(f'rm -rf {visual_dir}/*')
for i in ['match','miss']:
    os.makedirs(os.path.join(visual_dir,i),exist_ok=True)
    for j in ['0','1']:
        os.makedirs(os.path.join(visual_dir,i,j),exist_ok=True)
        

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.model_dir,'ridgeSegmentation',f"{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.model_dir,'ridgeSegmentation',f"{args.configs['save_name']}")))
model.eval()


# Test the model and save visualizations
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD
                )])
begin=time.time()
predict=[]
labels=[]

val_list_postive=[]
val_list_negtive=[]
val_list=[]
with torch.no_grad():
    for image_name in data_dict:
        mask=Image.open(data_dict[image_name]['mask_path']).resize((1600,1200),resample=Image.Resampling.BILINEAR)
        mask=np.array(mask)
        mask[mask>0]=1
    
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        output_img=F.interpolate(output_img,(1200,1600), mode='nearest')
        max_val=float(torch.max(output_img))
        val_list.append(max_val)
        output_img=output_img*mask
        
        if data['stage']>0:
            tar=1
            val_list_postive.append(max_val)
        else:
            tar=0
            val_list_negtive.append(max_val)
        if (max_val>=0.5):
            pred=1
        else:
            pred=0
        visual=True
        if args.visual_miss and tar!=pred:
            save_path=os.path.join(visual_dir,'miss',str(tar),image_name)
        elif args.visual_match and tar==pred:
            save_path=os.path.join(visual_dir,'match',str(tar),image_name)
        else:
            visual=False # do not visual
        if visual:
            visual_mask(data['image_path'],
                    output_img,
                    text_left=f"label:  {int(tar)}",
                    save_path=save_path)
        if pred==1:
            maxval,pred_point=k_max_values_and_indices(output_img.squeeze(),args.configs['ridge_Seg_number'],r=60,threshold=0.3)
            value_list=[]
            point_list=[]
            for value in maxval:
                value=round(float(value),2)
                value_list.append(value)
            for y,x in pred_point:
                point_list.append([int(x),int(y)])
            data_dict[image_name]['ridge_seg']={
                # "ridge_seg_path":ridge_seg_path,
                "value_list":value_list,
                "point_list":point_list,
                "orignal_weight":1600,
                "orignal_height":1200,
                'max_val':max_val,
                "sample_number":args.configs['ridge_Seg_number'],
                "sample_interval":60
            }
        else:
            data_dict[image_name]['ridge_seg']={
                'max_val':max_val
            }
        labels.append(tar)
        predict.append(pred)

acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
recall=recall_score(labels,predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")

# with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
with open('test.json','w') as f:
    json.dump(data_dict,f)