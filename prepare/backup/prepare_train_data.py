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

def extract_one_tar_file(tar_name, extract_to_folder,progress_queue):
    """解压单个 .tar 文件"""
    try:
        if tar_name.endswith('.tar'):
            with tarfile.open(tar_name, 'r') as tar:
                tar.extractall(path=extract_to_folder)
            progress_queue.put(1)
    except Exception as e:
        print(f"Error extracting {tar_name}: {e}")


def extract_all_tar_files_mul(tar_data_list_path, extract_to_folder, max_processes=60):
    """使用多进程解压多个 .tar 文件"""
    # 确保目标文件夹存在
    os.makedirs(extract_to_folder, exist_ok=True)

    # 读取 .tar 文件路径
    with open(tar_data_list_path, 'r', encoding='utf-8') as fp:
        tar_paths = [line.strip() for line in fp if 'train_l' in line]

    # 使用进程池并行解压
    # 使用Manager创建共享进度值
    with Manager() as manager:
        progress_queue = manager.Queue()
        with tqdm(total=len(tar_paths), desc="解压进度", position=0, ncols=100) as pbar:
            with ProcessPoolExecutor(max_processes) as executor:
                futures=[executor.submit(extract_one_tar_file,tar_name,extract_to_folder,progress_queue) for tar_name in tar_paths]
                completed_files=0
                for future in as_completed(futures):
                    future.result()
                    while not progress_queue.empty():
                        completed_files += progress_queue.get()
                        pbar.n = completed_files  # 更新进度条的当前值
                        pbar.last_print_n = completed_files  # 更新进度条的显示
                        pbar.update(0)  # 强制刷新进度条



#label.json file updated for audio 'duration' addition
def qa_json(input_label,output_label):
    with open(input_label,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_json=defaultdict(list)
    for k in data:
        for item in tqdm(data[k]):
            item['path']=item['path']+'.wav'
            duration=librosa.get_duration(path=item['path'])
            item['duration']=duration
            new_json[k].append(item)
    with open(output_label,'w',encoding='utf-8') as f:
        json.dump(new_json,f,ensure_ascii=False,indent=2)
            
        
    # audio
    # duration=librosa.get_duration(path=audio_path)
    
    # secret=[{'audio_name':name[:-4]+'.wav','path':os.path.join(input_dir,name[:-4]),'prompt':asr_question,'asr_label':asr} for name,asr in asr_dict.items()]
    # return secret



#train data from secret machine about 100 hours
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
                fp.write(file+' '+temp+'\n')
                fp.flush()

            audio_name=file[:-4]+'.wav'
            path=os.path.join(input_dir,audio_name)
            asr_label=temp
            duration=librosa.get_duration(path=path)
            # dict_item={'audio_name':audio_name,'path':path,'prompt':asr_question,'asr_label':asr_label}
            dict_item={'audio_name':audio_name,'path':path,'prompt':asr_question,'asr_label':asr_label,'duration':duration}
            final_dict.append(dict_item)
    with open(json_path,'w',encoding='utf-8') as f:
        json.dump(final_dict,f,ensure_ascii=False,indent=2)
   


def qa_pair_asr(asr_label_path_list,output_path,data_list):

    final_dict={}
    label_list=[]
    for asr_label_path in asr_label_path_list:
        with open(asr_label_path,'r',encoding='utf-8') as f:
            label={line.split(' ',maxsplit=1)[0].strip():line.split(' ',maxsplit=1)[1].strip() for line in f}
        label_list.append(label)

    for index,source in enumerate(data_list):
        temp=[{'audio_name':name,'prompt':asr_question,'asr_label':asr} for name,asr in label_list[index].items()]
        final_dict[source]=temp
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(final_dict,f,ensure_ascii=False,indent=2)



def json_with_duration(json_path,output_json):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data=[]
    for item in tqdm(data):
        path=item['path']
        duration=librosa.get_duration(path=path)
        item['duration']=duration
        new_data.append(item)
    with open(output_json,'w',encoding='utf-8') as f:
        data=json.dump(new_data,f,ensure_ascii=False,indent=2)


def qa_pair_asr_with_duration(asr_label_path_list,output_path,data_list,source_path):
    with open(source_path,'r',encoding='utf-8') as f:
        source_dict=json.load(f)
    final_dict={}
    label_list=[]
    for asr_label_path in asr_label_path_list:
        with open(asr_label_path,'r',encoding='utf-8') as f:
            label={line.split(' ',maxsplit=1)[0].strip():line.split(' ',maxsplit=1)[1].strip() for line in f}
        label_list.append(label)

    for index,source in enumerate(tqdm(data_list)):
        temp=[{'audio_name':name,'path':os.path.join(source_dict[source],name),'prompt':asr_question,'asr_label':asr,'duration':librosa.get_duration(path=os.path.join(source_dict[source],name))} for name,asr in tqdm(label_list[index].items())]
        final_dict[source]=temp
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(final_dict,f,ensure_ascii=False,indent=2)

def qa_pair_asr_with_duration_without_source(audio_dir,asr_label_path,output_path):
    with open(asr_label_path,'r',encoding='utf-8') as f:
        label_list=[{line.split('\t',maxsplit=1)[0].strip():line.split('\t',maxsplit=1)[1].strip()} for line in f]

    temp=[{'audio_name':name,'path':os.path.join(audio_dir,name),'prompt':asr_question,'asr_label':asr,'duration':librosa.get_duration(path=os.path.join(audio_dir,name))} for item in tqdm(label_list) for name,asr in item.items()]

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(temp,f,ensure_ascii=False,indent=2)

def qa_pair_asr_with_duration_without_source_ge1_from_json(json_path,output_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data=[]
    for item in data:
        if item['duration']>=1:
            new_data.append(item)
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)




def qa_pair_asrcor(asr_label_path_list,online_asr_label_path_list,output_path,data_list):
    final_dict={}
    label_list=[]
    online_asr_list=[]
    for asr_label_path in asr_label_path_list:
        with open(asr_label_path,'r',encoding='utf-8') as f:
            label={line.split(' ',maxsplit=1)[0].strip():line.split(' ',maxsplit=1)[1].strip() for line in f}
        label_list.append(label)

    for online_asr_label_path in online_asr_label_path_list:
        with open(online_asr_label_path,'r',encoding='utf-8') as f:
            label1={line.split(' ',maxsplit=1)[0].strip():line.split(' ',maxsplit=1)[1].strip() for line in f}
        online_asr_list.append(label1)

    for index,source in enumerate(data_list):
        temp=[{'audio_name':name,'prompt':asrcor_question.format(text=online_asr_list[index][name]),'asr_label':asr} for name,asr in label_list[index].items()]
        final_dict[source]=temp

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(final_dict,f,ensure_ascii=False,indent=2)

def qa_pair_asrcor_without_source(audio_dir,asr_label_path,online_asr_label_path,output_json_path):
    with open(asr_label_path,'r',encoding='utf-8') as f:
        asr_label={line.split(' ',maxsplit=1)[0].strip():line.split(' ',maxsplit=1)[1].strip() for line in f}

    with open(online_asr_label_path,'r',encoding='utf-8') as f:
        asrcor_label={line.split(' ',maxsplit=1)[0].strip():line.split(' ',maxsplit=1)[1].strip() for line in f}

    temp=[{'audio_name':name,'path':os.path.join(audio_dir,name),'prompt':asrcor_question.format(text=asrcor_label[name]),'asr_label':asr,'duration':librosa.get_duration(path=os.path.join(audio_dir,name))} for name,asr in tqdm(asr_label.items())]

    with open(output_json_path,'w',encoding='utf-8') as f:
        json.dump(temp,f,ensure_ascii=False,indent=2)


def model_answer_in_json(label_path,model_ans_txt,output_path):
    with open(label_path,'r',encoding='utf-8') as f:
        json_data=json.load(f)
    model_asr={}
    with open(model_ans_txt,'r',encoding='utf-8') as f:
        for line in f:
            items=line.split()
            model_asr[items[0]]=items[1] if len(items)>=2 else ''

    for item in json_data:
        name=item['name'][:-4] if item['name'].endswith('.wav') else item['name']
        item['model_answer']=model_asr[name]

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(json_data,f,ensure_ascii=False,indent=2)


#从100h数据抽取用于 质量检验 5->10->20->50
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


def sample_data_without_source(json_path,sampled_json_path,rate=0.05):
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



def remove_long_or_short(json_path,short_path,long_path,normal_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    short_data=[]
    long_data=[]
    normal_data=[]
    for item in data["xincheng_merge_sft_data_asr"]:
        if item['duration']>=1:
            short_data.append(item)
        if item['duration']<=30:
            long_data.append(item)
        normal_data.append(item)
    with open(short_path,'w',encoding='utf-8') as f1, open(long_path,'w',encoding='utf-8') as f2,open(normal_path,'w',encoding='utf-8') as f3:
        json.dump(short_data,f1,ensure_ascii=False,indent=2)
        json.dump(long_data,f2,ensure_ascii=False,indent=2)
        json.dump(normal_data,f3,ensure_ascii=False,indent=2)


def json2text(json_path,text_path):
    fp=open(text_path,'w',encoding='utf-8')
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
        print(len(data))
    for item in data:
        fp.write(item['name']+' '+item['label']+'\n')
        fp.flush()


#分割评测集对小于s和大于1s的数据的结果
def split_res(json_path,model_txt_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    json_root=os.path.dirname(json_path)
    output_label_s1_path=os.path.join(json_root,'label_cer_s1.json')
    output_label_g1_path=os.path.join(json_root,'label_cer_g1.json')
    label_s1=open(output_label_s1_path,'w',encoding='utf-8')
    label_g1=open(output_label_g1_path,'w',encoding='utf-8')
    s1_name=[]
    for item in data:
        if item['duration']<1:
            s1_name.append(item['name'])
            label_s1.write(item['name']+' '+item['label']+'\n')
        else:
            label_g1.write(item['name']+' '+item['label']+'\n')
    output_root=os.path.dirname(model_txt_path)
    output_txt_s1_path=os.path.join(output_root,'secret_data_qwen2aduio_asr_s1.txt')
    output_txt_g1_path=os.path.join(output_root,'secret_data_qwen2aduio_asr_g1.txt')
    model_s1=open(output_txt_s1_path,'w',encoding='utf-8')
    model_g1=open(output_txt_g1_path,'w',encoding='utf-8')
    with open(model_txt_path,'r',encoding='utf-8') as f1:
        for line in f1:
            line=line.strip()
            if not line: continue
            name=line.split()[0].strip()
            asr=line.split()[1].strip() if len(line.split())>1 else ''
            if name in s1_name:
                model_s1.write(name+' '+asr+'\n')
            else:
                model_g1.write(name+' '+asr+'\n')

def reformat_asr_txt(txt_path):
    new_data=[]
    with open(txt_path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                new_data.append(line)
    with open(txt_path,'w',encoding='utf-8') as f:
        for line in new_data:
            f.write(line+'\n')
            f.flush()



#过滤掉模糊的bench，输出过滤后清晰的bench
def clean_bench(label_txt,clean_label_txt):
    fp=open(clean_label_txt,'w',encoding='utf-8')
    with open(label_txt,'r',encoding='utf-8') as f:
        for line in f:
            items=line.split()
            name=items[0].strip()
            asr=items[1].strip()
            if not asr or '~' in asr:
                continue
            fp.write(name+' '+asr+'\n')
            fp.flush()


#根据online asr的结果，筛掉模糊部分，输出筛选后的asr结果
def anstxt_filter2_txt(txt_path,clean_label,clean_txt):
    clean_name_list=[]
    with open(clean_label,'r',encoding='utf-8') as f:
        for line in f:
            name=line.split()[0].strip()
            clean_name_list.append(name)
    
    clean_data={}
    with open(txt_path,'r',encoding='utf-8') as f:
        for line in f:
            items=line.split()
            name=items[0].strip()
            asr=items[1].strip() if len(items)>1 else ''
            clean_data[name]=asr

    with open(clean_txt,'w',encoding='utf-8') as f:
        for name,asr in clean_data.items():
            if name in clean_name_list:
                f.write(name+' '+asr+'\n')
                f.flush()

#根据原来bench的json，删除模糊部分，输出过滤后的bench
def txt_filter2json(json_path,clean_label,clean_json):
    clean_name_list=[]
    with open(clean_label,'r',encoding='utf-8') as f:
        for line in f:
            name=line.split()[0].strip()
            clean_name_list.append(name)
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data=[]
    for item in data:
        if item['name'] in clean_name_list:
            new_data.append(item)
    with open(clean_json,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)

#label.json to label.txt
def json2txt(json_path,txt_path):
    fp=open(txt_path,'w',encoding='utf-8')
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    for item in data:
        fp.write(item['name']+'\t'+item['label']+'\n')
        fp.flush()


def filter_for_g1_audios_from_json(json_path,output_json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data=[]
    for item in data:
        if item['duration']>=1:
            new_data.append(item)
    with open(output_json_path,'w',encoding='utf-8') as f:
        json.dump(new_data,f,indent=2,ensure_ascii=False)

#检测label是否有遗漏
def check(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    count=0
    for item in data:
        if not item['asr_label'] or item['asr_label']==' ':
            print(item)
            count+=1
    print(count)
    



if __name__=='__main__':
    # # data_list=['xincheng_sft_data_asr','xincheng_merge_sft_data_asr']
    # data_list=['xincheng_merge_sft_data_asr','xincheng_merge_sft_data_asr_new1']
    # # # data_list_secret=['secret_100h']
    # # # asr_label_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_rg_result1.txt'
    # asr_label_path1='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge1.txt'
    # asr_label_path2='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_label_new1.txt'
    # # # asr_merge_secret_label_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_100h_merge_label.txt'

    # # # asr_online_label_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_result1.txt'
    # # # asr_online_label_path1='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_oneline_asr.txt'

    # asr_label_path_list=[asr_label_path1,asr_label_path2]
    # # # asr_label_path_list=[asr_merge_secret_label_path]
    # # # online_asr_label_path_list=[asr_online_label_path,asr_online_label_path1]

    # # # output_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_merge_secret_audio_asr.json'
    # # # qa_pair_asr(asr_label_path_list,output_path,data_list_secret)

    # # # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_audio_asr_new1.json'
    # source_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/train/train_audio_dirs.json'
    # # # # # qa_pair_asr_with_duration(asr_label_path_list,output_path,data_list,source_path)

    # data_list=['xincheng_merge_sft_data_asr_0_30']
    # secret_label_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_label_0_30.txt'
    # asr_label_path_list=[secret_label_path]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_0_30.json'
    # qa_pair_asr_with_duration(asr_label_path_list,output_path,data_list,source_path)

    # classes=['drunk','magic','violence']
    # input_root='/nfs/dataset-411-391/guoruizheng'
    # for cls in tqdm(classes):
    #     input_json_file_path=os.path.join(input_root,cls,'small_asr.json')
    #     output_json=os.path.join(input_root,cls,'small_asr_with_duraion.json')
    #     json_with_duration(input_json_file_path,output_json)

    # root='/nfs/dataset-411-391/guoruizheng'
    # experiment='Qwen2Audio-Train-secret-100h-merge-5epoch-4batch-Test'
    # for cls in ['drunk','magic','violence']:
    #     label_path=os.path.join(root,cls,'small_asr_with_duraion.json')
    #     model_ans_txt=os.path.join(root,'results',experiment,cls,'secret_data_qwen2aduio_asr.txt')
    #     output_path=os.path.join(root,'results',experiment,cls,'secret_data_qwen2aduio_asr_model_answer.txt')
    #     model_answer_in_json(label_path,model_ans_txt,output_path)

    # json_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_merge_new11_secret_audio_asr.json'
    # sampled_json_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/sampled_50_train_merge_new11_secret_audio_asr.json'
    # sample_data(json_path,sampled_json_path,rate=0.5)

    #extrat files from tar
    # print(os.cpu_count())
    # tar_data_list_path='/nfs/dataset-411-391/wenet/examples/multi_cn/s0/data/train/open_filter_blank_zxbjjl_yueyu_national2.data'
    # extract_to_folder='/nfs/dataset-411-391/guoruizheng/wenetspeech/raw_data_completed'
    # # extract_all_tar_files(tar_data_list_path, extract_to_folder)
    # extract_all_tar_files_mul(tar_data_list_path,extract_to_folder)
    # /nfs/volume-242-2/dengh/Qwen2-audio/src/prepare/prepare_train_data.py

    # input_dir='/nfs/dataset-411-391/guoruizheng/wenetspeech/raw_data'
    # output_file='/nfs/dataset-411-391/guoruizheng/wenetspeech/wenetspeech_150_asr.txt'
    # json_path='/nfs/dataset-411-391/guoruizheng/wenetspeech/wenetspeech_150_asr.json'
    # qa_pair_secret(input_dir,output_file,json_path)

    # json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_audio_asr.json'
    # short_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_short.json'
    # long_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_long.json'
    # normal_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_completed.json'
    # remove_long_or_short(json_path,short_path,long_path,normal_path)

    # input_dir='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/nl0/all_wav'
    # asr_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_rg_result1.txt'
    # json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/xincheng_sft_data_asr.json'
    # qa_pair_asr_with_duration_without_source(input_dir,asr_path,json_path)


    # json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/xincheng_sft_data_asr.json'
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/xincheng_sft_data_ge1_asr.json'
    # qa_pair_asr_with_duration_without_source_ge1_from_json(json_path,output_path)

    # json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench.json'
    # text_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer.json'
    # json2text(json_path,text_path)


    # json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench.json'
    # model_txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/Online-ASR/secret_data_qwen2aduio_asr.txt'
    # split_res(json_path ,model_txt_path)

    # reformat_asr_txt('/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/Qwen2Audio-Test/secret_data_qwen2aduio_asr.txt')
    # label_json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench.json'
    # model_list=['Online-ASR','Qwen2Audio-Test','Qwen2Audio-Train-merge-5epoch-4batch-Test','Qwen2Audio-Train-merge-new1-5epoch-4batch-Test']
    # res_root='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results'
    # output_csv='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/res_col.csv'
    
    # res_col(label_json_path, model_list, res_root,output_csv)

    # audio_dir='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_1_30'
    # asr_label_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_label_0_30.txt'
    # online_asr_label_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_asrcor_1_30.txt'
    # output_json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_merge_data_asrcor_1_30.json'

    # # qa_pair_asrcor_without_source(audio_dir,asr_label_path,online_asr_label_path,output_json_path)
    # label_txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer.json'
    # clean_label_txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_clean.txt'
    # clean_bench(label_txt_path,clean_label_txt_path)

    # json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench.json'
    # clean_label_txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_clean.txt'
    # clean_json='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json'
    # txt_filter2json(json_path,clean_label_txt_path,clean_json)

    # txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/Qwen2Audio-Test/secret_data_qwen2aduio_asr.txt'
    # clean_label_txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_clean.txt'
    # clean_txt='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/Qwen2Audio-Test/secret_data_qwen2aduio_asr_clean.txt'
    # anstxt_filter2_txt(txt_path,clean_label_txt_path,clean_txt)

    # reformat_asr_txt('/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/results/Qwen2Audio-Train-merge-new1-5epoch-4batch-freeze-llm+projector-Test/secret_data_qwen2aduio_asr_clean.txt')
    # json_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench_clean.json'
    # txt_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/label_cer_clean.txt'
    # json2txt(json_path,txt_path)

    json_path='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr_g1.json'
    # output_json_path='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr_g1.json'
    # filter_for_g1_audios_from_json(json_path,output_json_path)
    check(json_path)

