import json
import os
import librosa
from tqdm import tqdm 
import random
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
from multiprocessing import Manager
import pandas as pd


asr_question="""<|audio_bos|><|AUDIO|><|audio_eos|>
你需要完成一个中文自动语音识别任务，将输入的音频数据转换成相应的文本。请根据音频中的语音内容准确生成文本输出,直接输出文本。
要求：
1.生成的文本必须尽可能准确地反映音频中的内容。
2.文本输出应以正常的文本格式提供，无需添加额外的标签或格式。
3.请直接输出文本，不要随意添加任何额外的说明。
"""


asrcor_question="""<|audio_bos|><|AUDIO|><|audio_eos|>
请根据音频中的语音内容对以下文本 <text> 进行纠错。你的目标是对 <text> 进行必要的修改，以确保其准确反映音频内容。
要求: 
1.如果 <text> 已经与音频内容一致，则无需进行修改，输出保持不变。
2.请直接输出最终的 ASR 文本，不要随意添加任何额外的说明。
3.如果音频为空或无任何内容, 则输出' '。
<text>: {text}
"""

qwen2audio_asrcor_question="""<|audio_bos|><|AUDIO|><|audio_eos|>
请根据音频中的语音内容对以下文本 <text> 进行纠错。你的目标是对 <text> 进行必要的修改，以确保其准确反映音频内容。
要求: 
1.如果 <text> 已经与音频内容一致，则无需进行修改，输出保持不变。
2.请直接输出最终的 ASR 文本，不要随意添加任何额外的说明, 如："纠正就的文本是"。
3.如果音频为空或无任何内容, 则输出' '。
<text>: {text}
"""


def extract_all_tar_files(tar_data_list_path, extract_to_folder):
    # 确保目标文件夹存在
    os.makedirs(extract_to_folder, exist_ok=True)
    
    fp=open(tar_data_list_path,'r',encoding='utf-8')
    tar_paths=[line.strip() for line in fp if 'train_l' in line]

    # 遍历文件夹中的所有文件
    for tar_name in tqdm(tar_paths):
        # 检查文件是否为 .tar 文件
        if tar_name.endswith('.tar'):
            # 打开并解压 .tar 文件
            with tarfile.open(tar_name, 'r') as tar:
                tar.extractall(path=extract_to_folder)


#train_data from secret machine about 100 hours
#both .wav and .txt files in input_dir
def qa_pair_secret(input_dir,asr_path,json_path):
    final_dict=[]
    asr_dict={}
    fp=open(asr_path,'w',encoding='utf-8')
    for file in tqdm(os.listdir(input_dir)):
        if file.endswith('txt'):
            with open(os.path.join(input_dir,file),'r',encoding='utf-8') as f:
                temp=''
                for line in f:
                    temp+=line.strip()
                asr_dict[file]=temp
                fp.write(file+'\t'+temp+'\n')
                fp.flush()

            audio_name=file[:-4]+'.wav'
            path=os.path.join(input_dir,audio_name)
            asr_label=temp
            duration=librosa.get_duration(path=path)
            dict_item={'audio_name':audio_name,'path':path,'prompt':asr_question,'asr_label':asr_label,'duration':duration}
            final_dict.append(dict_item)
    with open(json_path,'w',encoding='utf-8') as f:
        json.dump(final_dict,f,ensure_ascii=False,indent=2)
   


def qa_pair_asrcor(audio_dir,asr_label_path,online_asr_label_path,output_json_path):
    with open(asr_label_path,'r',encoding='utf-8') as f:
        asr_label={line.split('\t',maxsplit=1)[0].strip():line.split('\t',maxsplit=1)[1].strip() for line in f}

    with open(online_asr_label_path,'r',encoding='utf-8') as f:
        asrcor_label={line.split('\t',maxsplit=1)[0].strip():line.split('\t',maxsplit=1)[1].strip() for line in f}

    temp=[{'audio_name':name,'path':os.path.join(audio_dir,name),'prompt':asrcor_question.format(text=asrcor_label[name]),'asr_label':asr,'duration':librosa.get_duration(path=os.path.join(audio_dir,name))} for name,asr in tqdm(asr_label.items())]

    with open(output_json_path,'w',encoding='utf-8') as f:
        json.dump(temp,f,ensure_ascii=False,indent=2)


def sample_data(json_path,sampled_json_path,rate=0.05):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data={}
    for source in data:
        temp=[]
        random.shuffle(data[source])
        sampled_count=int(rate*len(data[source]))
        for item in data[source][:sampled_count]:
            temp.append(item)
        new_data[source]=temp
        
    with open(sampled_json_path,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)


def json2text(json_path,text_path):
    fp=open(text_path,'w',encoding='utf-8')
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        print(len(data))
    for item in data:
        fp.write(item['name']+'\t'+item['label']+'\n')
        fp.flush()





if __name__=='__main__':
    pass