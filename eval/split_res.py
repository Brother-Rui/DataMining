from io import BytesIO
import os
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import json
import re

# def split_res(json_path,model_txt_path):
#     with open(json_path,'r',encoding='utf-8') as f:
#         data=json.load(f)
#     s1_name=[]
#     for item in data:
#         if item['duration']<1:
#             s1_name.append(item['name'])
#     output_root=os.path.dirname(model_txt_path)
#     output_txt_s1_path=os.path.join(output_root,'secret_data_qwen2aduio_asr_s1.txt')
#     output_txt_g1_path=os.path.join(output_root,'secret_data_qwen2aduio_asr_g1.txt')
#     model_s1=open(output_txt_s1_path,'w',encoding='utf-8')
#     model_g1=open(output_txt_g1_path,'w',encoding='utf-8')
#     with open(model_txt_path,'r',encoding='utf-8') as f1:
#         nan_count=0
#         for line in f1:
#             line=line.strip()
#             if not line: continue
#             name=line.split()[0].strip()
#             asr=line.split()[1].strip() if len(line.split())>1 else ''
#             if not asr:nan_count+=1
#             if name in s1_name:
#                 model_s1.write(name+' '+asr+'\n')
#             else:
#                 model_g1.write(name+' '+asr+'\n')
#         print(f'{os.path.basename(output_root)}:{nan_count}')



#分割评测集对小于s和大于1s的数据的结果
def split_res_clean(json_path,model_txt_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    # #对label进行划分
    # json_root=os.path.dirname(json_path)
    # output_label_s1_path=os.path.join(json_root,'label_cer_s1_clean.txt')
    # output_label_g1_path=os.path.join(json_root,'label_cer_g1_clean.txt')
    # label_s1=open(output_label_s1_path,'w',encoding='utf-8')
    # label_g1=open(output_label_g1_path,'w',encoding='utf-8')
    
    # #对model answer进行划分
    s1_name=[]
    for item in data:
        if item['duration']<1:
            s1_name.append(item['name'])
    #         label_s1.write(item['name']+' '+item['label']+'\n')
    #     else:
    #         label_g1.write(item['name']+' '+item['label']+'\n')
    output_root=os.path.dirname(model_txt_path)
    output_txt_s1_path=os.path.join(output_root,'secret_data_qwen2aduio_asr_s1_clean.txt')
    output_txt_g1_path=os.path.join(output_root,'secret_data_qwen2aduio_asr_g1_clean.txt')
    model_s1=open(output_txt_s1_path,'w',encoding='utf-8')
    model_g1=open(output_txt_g1_path,'w',encoding='utf-8')
    with open(model_txt_path,'r',encoding='utf-8') as f1:
        nan_count=0
        for line in f1:
            line=line.strip()
            # if not line: continue
            name=line.split('\t')[0].strip()
            asr=line.split('\t')[1].strip() if len(line.split('\t'))>1 else ''
            if not asr :nan_count+=1
            if name in s1_name:
                model_s1.write(name+'\t'+asr+'\n')
            else:
                model_g1.write(name+'\t'+asr+'\n')
    print(f'Nan of {os.path.basename(output_root)}:{nan_count}')



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--json_path',type=str)
    parser.add_argument('--model_txt_path',type=str)
    args=parser.parse_args()
    split_res_clean(args.json_path,args.model_txt_path)
