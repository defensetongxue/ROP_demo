import json,os
from config import get_config
args=get_config()
data_path=args.data_path
with open(os.path.join(data_path,'annotations.json'),'r') as f:
    data_dict=json.load(f)
with  open(args.handcraft_path,'r') as f:
    handcraft=json.load(f)
for  image_name in handcraft:
    data_dict[image_name]['stage']=handcraft[image_name]['stage']
    data_dict[image_name]['zone']=handcraft[image_name]['zone']
with open(os.path.join(data_path,'annotations.json'),'w') as f:
    json.dump(data_dict,f)
    