import json
import os
from tqdm import tqdm 
from collections import defaultdict
import pandas as pd


#results collection
def res_col(label_json_path, model_list, res_root,output_csv,url_prefix='https://easyai.xiaojukeji.com/DEDataSetMgtView?type=view&setId=45901&rowName='):
    with open(label_json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    columns=['Key','duration','label',*model_list,'url']
    model_res=defaultdict(dict)
    for model in model_list:
        with open(os.path.join(res_root,model,'secret_data_qwen2aduio_asr_clean.txt'),'r',encoding='utf-8') as f:
            for line in f:
                temp={}
                temp_res=line.split('\t')
                # if not temp_res:continue
                name=temp_res[0].strip()
                asr=temp_res[1].strip() if len(temp_res)>1 else ''
                temp[name]=asr
                model_res[model][name]=asr
    df=pd.DataFrame(columns=columns)
    for item in data:
        temp_list=[]
        key=item['name']
        value=item['label']
        duration=item['duration']
        temp_list=[key, duration,value]
        for model in model_list:
            temp_list.append(model_res[model][key])
        url_path=url_prefix+key+'.wav'
        temp_list.append(url_path)
        new_row={}
        for id,col in enumerate(columns):
            new_row[col]=temp_list[id]
        new_row=pd.DataFrame(new_row,index=[0])
        df=pd.concat([df,new_row],ignore_index=True)
    df.to_csv(output_csv,index=False,encoding='utf-8')
    # df.to_excel(output_csv, index=False,engine='xlsxwriter')


if __name__=='__main__':
    #clean bench
    label_json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json'
    model_list=['Qwen2Audio-Train-100h_merge_without_insertion-2epoch-4batch-train-first_last2+projector-Test','Qwen2Audio-Train-100h_merge_without_insertion-1epoch-4batch-train-first_last2_attn+projector']
    res_root='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results'
    output_csv='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/res_csv/res_col_freeze_1_15.csv'
    
    res_col(label_json_path, model_list, res_root,output_csv)

    