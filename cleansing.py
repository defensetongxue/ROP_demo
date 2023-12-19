import os,json
from config import get_config
args=get_config()
data_path=args.data_path

image_list=sorted(os.listdir(os.path.join(data_path,'images')),lambda x: int(x.split('.')[0]))
annotations={}
for image_name in image_list:
    annotations[image_name]={
        "image_path":os.path.join(data_path,'images',image_name),
        "stage":0,
        "zone":0,
        "plus":0,
    }
if os.path.exists(args.handcraft_path):
    with  open(args.handcraft_path,'r') as f:
        handcraft=json.load(f)
    for  image_name in handcraft:
        annotations[image_name]['stage']=handcraft[image_name]['stage']
        annotations[image_name]['zone']=handcraft[image_name]['zone']

with open(os.path.join(data_path,'annotations.json'),'w') as f:
    json.dump(annotations,f)
    