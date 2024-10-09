import os
import json
# train_path = '/home/zhoujh/wav/train/'
else_path = '/home/zhoujh/wav/else/'
nv_path = '/home/zhoujh/speech2SQL/data/nvbench/NVBench.json'
# nv_data = []
with open(nv_path) as inf:
        nv_data = json.load(inf)
        # data = lower_keys(data)
        # nv_data += data#[:10000]
        # print('total data:', len(nv_data))
# file_list = os.listdir(train_path)
file_list = os.listdir(else_path)
whole_data = []
for file_name in file_list:
    # 提取id1和id2
    data = {}
    id1, id2= file_name.replace('_result.wav', '').rsplit('_',1)
    for nv in nv_data:
        #   if id1==nv[]
        #   print('')
        if id1==nv:
            data = nv_data[nv]
            data['id1']=id1
            data['id2']=int(id2)
            whole_data.append(data)
            break
# whole_data.to_json('tts_pretrain.json', orient = 'records', indent = 1)
json_str = json.dumps(whole_data)
with open('tts_pretrain_else.json','w') as file:
    file.write(json_str)