import json
import os
import torch
from config import get_config
from torchvision import transforms
from util.tools import get_instance
from util.ridge_segment import k_max_values_and_indices
import ridgeSegmentModel as models
from PIL import Image
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score,recall_score
import numpy as np
# Parse arguments
torch.manual_seed(0)
np.random.seed(0)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
args = get_config()
with open(args.ridge_seg_cfg,'r') as f:
    args.configs=json.load(f)
    

# Test the model and save visualizations
with open(os.path.join(args.data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
# Init the result file to store the pytorch model and other mid-result
result_path = args.result_path
os.makedirs(result_path,exist_ok=True)
print(f"the mid-result and the pytorch model will be stored in {result_path}")

# Create the model and criterion
model = get_instance(models, args.configs['model']['name'],args.configs['model'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.load_state_dict(
    torch.load(os.path.join(args.model_dir,'ridgeSegmentation',f"{args.configs['save_name']}")))
print("load the checkpoint in {}".format(os.path.join(args.model_dir,'ridgeSegmentation',f"{args.configs['save_name']}")))
model.eval()


img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,std=IMAGENET_DEFAULT_STD
                )])
predict=[]
labels=[]
model_predit={}
save_all_visual=True
save_all_dir=os.path.join(args.data_path,'ridge_seg')
os.makedirs(save_all_dir,exist_ok=True)
with open(os.path.join(args.data_path,'split','clr.json')) as f:
    split_list=json.load(f)['test']
    
with torch.no_grad():
    for image_name in split_list:
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
        output_img=output_img*mask
        max_val=float(torch.max(output_img))
        
        if data['stage']>0:
            tar=1
        else:
            tar=0
        if (max_val>0.5):
            pred=1
        else:
            pred=0
        
        if max_val>0.5:
            # Construct the file path for saving the image
            ridge_seg_path = os.path.join(save_all_dir,image_name)
            
            # Squeeze the tensor to remove any extra dimensions
            output_img = output_img.squeeze()
            
            # Convert the tensor to a PIL image
            # Assuming the tensor is in the range [0, 1)
            output_img_pil = Image.fromarray((output_img.numpy() *255).astype('uint8'))
            
            # Save the image
            output_img_pil.save(ridge_seg_path)
                
            
            data_dict[image_name]['ridge_seg']={
                "ridge_seg_path":ridge_seg_path,
                "orignal_weight":1600,
                "orignal_height":1200,
                'max_val':max_val
                
            }
        else:
            data_dict[image_name]['ridge_seg']={
                'max_val':max_val
            }
        labels.append(tar)
        predict.append(pred)
        # break
acc = accuracy_score(labels, predict)
auc = roc_auc_score(labels, predict)
recall=recall_score(labels,predict)
print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Recall: {recall:.4f}")

with open(os.path.join(args.data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)
