import os,json
from shutil import copy
data_path='../autodl-tmp/dataset_ROP/'
with open(os.path.join(data_path,'annotations.json')) as f:
    data_dict=json.load(f)
with open(os.path.join(data_path,'split','clr_1.json')) as f:
    used_data_dict=json.load(f)
    used_data=used_data_dict['train']+used_data_dict['val']+used_data_dict['test']
tar_dir='../autodl-tmp/release'
os.makedirs(tar_dir,exist_ok=True)
cnt={1:0,2:0,3:0}
for image_name in used_data:
    data=data_dict[image_name]
    if data['stage']>0 and data['ridge_seg']['max_val']>0.5 and 'ridge' in data:
        copy(data['image_path'],os.path.join(tar_dir,f"{str(data['stage'])}_{image_name}"))
        cnt[data['stage']]+=1
print(cnt)