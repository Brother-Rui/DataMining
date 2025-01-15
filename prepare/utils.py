import json
import jsonlines
from tqdm import tqdm
from collections import defaultdict
# import editdistance
import os 
import librosa
import matplotlib.pyplot as plt
import matplotlib.image as mping
import pandas as pd

asr_question="""<|audio_bos|><|AUDIO|><|audio_eos|>
你需要完成一个中文自动语音识别任务，将输入的音频数据转换成相应的文本。请根据音频中的语音内容准确生成文本输出,直接输出文本。
要求：
1.生成的文本必须尽可能准确地反映音频中的内容。
2.文本输出应以正常的文本格式提供，无需添加额外的标签或格式。
3.请直接输出文本，不要随意添加任何额外的说明。
"""



def bench_reformat(input_path,output_path):
    res=[]
    with jsonlines.open(input_path) as f:
        for obj in tqdm(f, desc="Processing files",total=2844):
            temp={}
            temp['name']=obj['key']
            temp['path']=obj['wav']
            temp['label']=obj['txt']
            temp['duration']=librosa.get_duration(path=temp['path'])
            res.append(temp)
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(res,f,ensure_ascii=False,indent=2)


#去掉source
def remove_source(json_path,json_path_new):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data=[]
    for source in data:
        new_data+=data[source]
    with open(json_path_new,'w',encoding='utf-8') as f:
        json.dump(new_data,f,indent=2,ensure_ascii=False)
    return json_path_new

#修正json文件的prompt
def reformat_json_prompt(json_path,json_path_new):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    for item in data:
        item['prompt']=asr_question
    with open(json_path_new,'w',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False)

    
    
    
        


if __name__=='__main__':
    pass


