import json,os
with open('../autodl-tmp/dataset_ROP/split/1.json') as f:
    split_list=json.load(f)
cnt=0
for split in split_list:
    cnt+=len(split_list[split])
print(cnt)

test_list=os.listdir("../autodl-tmp/ROP_shen/images")
print(len(test_list))