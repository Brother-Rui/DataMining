
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



#from json_files
def count_audio_files_and_duration_from_json_without_source(json_path,duration_output_file,output_pic):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    durations=[]
    res={}
    for item in data:
        duration = item['duration']
        key=f'[{int(duration)},{int(duration)+1})'
        res[key]=res.get(key,0)+1
        durations.append(duration)

    sorted_keys=sorted(res.keys(),key=lambda x:int(x.split(',')[0][1:]))
    sorted_res={k:res[k] for k in sorted_keys}

    with open(duration_output_file,'w',encoding='utf-8') as f:
        json.dump(sorted_res,f,ensure_ascii=False,indent=2)

    num_audio_files = len(durations)
    total_duration = sum(durations)
    average_duration = total_duration / num_audio_files if num_audio_files > 0 else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    print(f'音频文件总数: {num_audio_files}')
    print(f'音频总时长: {total_duration:.2f} 秒')
    print(f'音频平均时长: {average_duration:.2f} 秒')
    print(f'最长音频时长: {max_duration:.2f} 秒')
    print(f'最短音频时长: {min_duration:.2f} 秒')
    
    # 绘制音频时长分布图
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title('Audio Durations Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Audio Files')
    plt.xlim(min_duration, min(60,max_duration))

    # plt.show()
    plt.savefig(output_pic)


#from json_files
def count_audio_files_and_duration_from_json_with_source(json_path,output_file,output_pic):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    durations=[]
    res={}
    for source in data:
        for item in data[source]:
            duration = item['duration']
            key=f'[{int(duration)},{int(duration)+1})'
            res[key]=res.get(key,0)+1
            durations.append(duration)

    sorted_keys=sorted(res.keys(),key=lambda x:int(x.split(',')[0][1:]))
    sorted_res={k:res[k] for k in sorted_keys}
        
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(sorted_res,f,ensure_ascii=False,indent=2)

    num_audio_files = len(durations)
    total_duration = sum(durations)
    average_duration = total_duration / num_audio_files if num_audio_files > 0 else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    print(f'音频文件总数: {num_audio_files}')
    print(f'音频总时长: {total_duration:.2f} 秒')
    print(f'音频平均时长: {average_duration:.2f} 秒')
    print(f'最长音频时长: {max_duration:.2f} 秒')
    print(f'最短音频时长: {min_duration:.2f} 秒')
    
    # 绘制音频时长分布图
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title('Audio Durations Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Audio Files')
    plt.xlim(min_duration, min(60,max_duration))

    # plt.show()
    plt.savefig(output_pic)



#count from audio
def count_audio_files_and_duration_from_audios(train_audio_dir,output_file,output_pic):
    durations=[]
    audios_path=[os.path.join(train_audio_dir,audio) for audio in os.listdir(train_audio_dir) if audio.endswith('.wav')]
    res={}
    for audio_path in tqdm(audios_path):               
        duration = librosa.get_duration(path=audio_path)
        key=f'[{int(duration)},{int(duration)+1})'
        res[key]=res.get(key,0)+1
        durations.append(duration)

    sorted_keys=sorted(res.keys(),key=lambda x:int(x.split(',')[0][1:]))
    sorted_res={k:res[k] for k in sorted_keys}

    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(sorted_res,f,ensure_ascii=False,indent=2)

    num_audio_files = len(durations)
    total_duration = sum(durations)
    average_duration = total_duration / num_audio_files if num_audio_files > 0 else 0
    max_duration = max(durations) if durations else 0
    min_duration = min(durations) if durations else 0

    print(f'音频文件总数: {num_audio_files}')
    print(f'音频总时长: {total_duration:.2f} 秒')
    print(f'音频平均时长: {average_duration:.2f} 秒')
    print(f'最长音频时长: {max_duration:.2f} 秒')
    print(f'最短音频时长: {min_duration:.2f} 秒')
    
    # 绘制音频时长分布图
    plt.hist(durations, bins=30, color='skyblue', edgecolor='black')
    plt.title('Audio Durations Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Audio Files')
    plt.xlim(min_duration, min(60,max_duration))

    # plt.show()
    plt.savefig(output_pic)



if __name__=='__main__':
    # json_path='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr1.json'
    # output_file='/nfs/dataset-411-391/guoruizheng/train_100h_info/train_100h_secret_audio_info/record.txt'
    # output_pic='/nfs/dataset-411-391/guoruizheng/train_100h_info/train_100h_secret_audio_info/record.png'
    json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/source_data/xincheng_merge_sft_data_asr_new1.json'
    output_file='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/data_info/xincheng_merge_sft_data_asr_new1/record.txt'
    output_pic='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/data_info/xincheng_merge_sft_data_asr_new1/record.txt'
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    count_audio_files_and_duration_from_json_without_source(json_path,output_file,output_pic)
