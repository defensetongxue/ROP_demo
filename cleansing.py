import os,json
from config import get_config
args=get_config()
data_path=args.data_path

image_list=sorted(os.listdir(os.path.join(data_path,'images')),key=lambda x: int(x.split('.')[0]))
annotations={}
split_list={'train':[],'val':[],'test':[]}
for image_name in image_list:
    annotations[image_name]={
        "image_path":os.path.join(data_path,'images',image_name),
        "stage":0,
        "zone":0,
        "plus":0,
    }
    split_list.append(image_name)

with open(os.path.join(data_path,'annotations.json'),'w') as f:
    json.dump(annotations,f)
os.makedirs(os.path.join(data_path,'split'),exist_ok=True)
with open(os.path.join(data_path,'split','all.json'),'w') as f:
    json.dump(split_list,f)
     