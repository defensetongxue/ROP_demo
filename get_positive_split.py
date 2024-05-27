import os,json
from config import get_config
data_path='../autodl-tmp/ROP_shen'
with open(os.path.join(data_path,'annotations.json')) as f:
    data_dict=json.load(f)

with open('./label_v1.json') as f:
    label_v1=json.load(f)
with open(os.path.join(data_path,'split','all.json')) as f:
    split_all=json.load(f)['test']

args=get_config()
with open(args.stage_cfg,'r') as f:
    args.configs=json.load(f)
positive_split={'test':[]}
cnt=0
for image_name in split_all:
    data=data_dict[image_name]
    if data['ridge_seg']["max_val"]<args.configs['judge_threshold'] or data_dict[image_name]['stage']==0:
        continue
    positive_split['test'].append(image_name)
    cnt+=1

print(cnt)