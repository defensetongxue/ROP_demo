import json
import os
import torch
from config import get_config
from torchvision import transforms
from util import get_instance,visual_mask
import ridgeSegmentModel as models
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score
import numpy as np
# Parse arguments
import time
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
os.makedirs(visual_dir+'/0/', exist_ok=True)
os.makedirs(visual_dir+'/1/', exist_ok=True)

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
        mask=Image.open(data_dict[image_name]['mask_path']).resize((800,600),resample=Image.Resampling.BILINEAR)
        mask=np.array(mask)
        mask[mask>0]=1
    
        data=data_dict[image_name]
        img = Image.open(data['enhanced_path'])
        img_tensor = img_transforms(img)
        
        img=img_tensor.unsqueeze(0).to(device)
        output_img = model(img).cpu()
        # Resize the output to the original image size
        
        output_img=torch.sigmoid(output_img)
        max_val=float(torch.max(output_img))
        val_list.append(max_val)
        output_img=F.interpolate(output_img,(600,800), mode='nearest')
        output_img=output_img*mask
        
        output_bin=torch.where(output_img>0.5,1,0).squeeze()
        if data['stage']>0:
            tar=1
            val_list_postive.append(max_val)
        else:
            tar=0
            val_list_negtive.append(max_val)
        if (torch.sum(output_bin)>=1):
            pred=1
        else:
            pred=0
        if pred!=tar:
            output_img=output_img.squeeze()
            if tar==0:
                visual_mask(data['image_path'],output_img,str(round(max_val,2)),save_path=os.path.join(visual_dir,'0',image_name))
            else:
                visual_mask(data['image_path'],output_img,str(round(max_val,2)),
                            save_path=os.path.join(visual_dir,'1',image_name))
        labels.append(tar)
        predict.append(pred)

acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
recall=recall_score(labels,predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")