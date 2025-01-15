import json
import os
from collections import defaultdict
from pydub import AudioSegment
from tqdm import tqdm
import numpy as np
from scipy.signal import butter, lfilter
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import Manager,cpu_count
import librosa

asr_question="""<|audio_bos|><|AUDIO|><|audio_eos|>
你需要完成一个中文自动语音识别任务，将输入的音频数据转换成相应的文本。请根据音频中的语音内容准确生成文本输出,直接输出文本。
要求：
1.生成的文本必须尽可能准确地反映音频中的内容。
2.文本输出应以正常的文本格式提供，无需添加额外的标签或格式。
3.请直接输出文本，不要随意添加任何额外的说明。
"""


def generate_engine_noise(duration_ms=1000, volume_db=-20, engine_frequency=50):
    """
    模拟发动机的轰隆隆噪声，主要为低频振动。
    
    :param duration_ms: 噪声持续时间（毫秒）。
    :param volume_db: 音量（dB）。
    :param engine_frequency: 模拟发动机频率（Hz）。
    :return: AudioSegment 对象。
    """
    sample_rate = 44100
    num_samples = int(sample_rate * (duration_ms / 1000))
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)

    # 生成低频正弦波，模拟发动机振动
    low_freq_wave = (np.sin(2 * np.pi * engine_frequency * t) * 32767).astype(np.int16)

    # 添加随机抖动，模拟不规则的发动机振动
    jitter = np.random.uniform(-0.2, 0.2, num_samples) * 32767
    engine_noise = (low_freq_wave + jitter).astype(np.int16)

    # 转换为 AudioSegment
    engine_audio = AudioSegment(
        engine_noise.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    ).apply_gain(volume_db)

    return engine_audio

def generate_wind_noise(duration_ms=1000, volume_db=-40):
    """
    模拟车内风噪，主要为高频噪声。
    
    :param duration_ms: 噪声持续时间（毫秒）。
    :param volume_db: 音量（dB）。
    :return: AudioSegment 对象。
    """
    sample_rate = 44100
    num_samples = int(sample_rate * (duration_ms / 1000))

    # 生成白噪声
    white_noise = np.random.uniform(-1, 1, num_samples) * 32767
    white_noise = white_noise.astype(np.int16)

    # 高通滤波器，提取高频成分
    def butter_highpass_filter(data, cutoff, fs, order=6):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, data)

    high_freq_noise = butter_highpass_filter(white_noise, cutoff=2000, fs=sample_rate)

    # 转换为 AudioSegment
    wind_audio = AudioSegment(
        high_freq_noise.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    ).apply_gain(volume_db)

    return wind_audio

#car noise
def generate_car_noise(duration_ms=1000, engine_volume_db=-50, wind_volume_db=-60):
    """
    生成综合的车内噪音，包括轰隆隆的发动机声和风噪。
    
    :param duration_ms: 噪声持续时间（毫秒）。
    :param engine_volume_db: 发动机噪音音量（dB）。
    :param wind_volume_db: 风噪音量（dB）。
    :return: AudioSegment 对象。
    """
    engine_noise = generate_engine_noise(duration_ms, volume_db=engine_volume_db)
    wind_noise = generate_wind_noise(duration_ms, volume_db=wind_volume_db)

    # 混合两种噪声
    car_noise = engine_noise.overlay(wind_noise)

    return car_noise

#silence
def generate_silence(duration_ms=1000):
    return AudioSegment.silent(duration=duration_ms)

def concatenate_audios_with_insert(wav_dir,audios_path, insert_segment,insert_flag=True):
    audios=[AudioSegment.from_file(os.path.join(wav_dir,audio_path)) for audio_path in audios_path]
    combined = AudioSegment.empty()
    for index,audio in enumerate(audios):
        combined += audio
        #插入间隔
        if insert_flag and index<len(audios)-1:
            combined += insert_segment
    return combined

# def merge_asr_audio(wav_asr_path,wav_dir,merge_wav_dir,merge_asr_path):
#     fp=open(wav_asr_path,'r',encoding='utf-8')
#     wav_asr=[[line.strip().split()[0],(line.strip().split()[1] if len(line.strip().split())>=2 else '')] for line in fp]
#     merge_dict=defaultdict(lambda:{'wav':[],'asr':[]})
#     for item in wav_asr:
#         wav=''
#         if item[0].endswith('.wav'):
#             wav=item[0]
#         elif item[0].endswith('.txt'):
#             wav=item[0][:-4]+'.wav'
#         else:
#             wav=item[0]+'.wav'
        
#         asr=item[1]
#         merge_dict[wav.split('.')[0]]['wav'].append(wav)
#         merge_dict[wav.split('.')[0]]['asr']+=[asr+' ']
#     insert_segment=generate_silence()
#     fp=open(merge_asr_path,'w')
#     for wav_name,item in tqdm(merge_dict.items()):
#         merge_audio=concatenate_audios_with_insert(wav_dir,item['wav'],insert_segment)
#         merge_audio.export(os.path.join(merge_wav_dir,(wav_name+'.wav')), format="wav")
#         fp.write(wav_name+'.wav'+' '+''.join(item['asr']).strip()+'\n')
#         fp.flush()


#merge audios with duration
def merge_asr_audio_with_duration(json_path,wav_dir,merge_wav_dir,merge_asr_path,insert_len=1):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    min_len=1
    max_len=30
    merge_dict=defaultdict(lambda:{'wav':[],'asr':[],'duration':[]})

    for source in data:
        same_periods=defaultdict(list)
        for item in data[source]:
            key=item['audio_name'].split('.')[0]
            same_periods[key].append(item)

        len_same_periods=defaultdict(float)
        for k,v in same_periods.items():
            len_same_periods[k]=sum([subitem['duration'] for subitem in v])+(len(same_periods)-1)*insert_len

        for name,durations in len_same_periods.items():
            if durations<min_len:continue
            elif min_len<=durations<=max_len:
                merge_dict[name]['wav']=[item['audio_name'] for item in same_periods[name]]
                merge_dict[name]['asr']=[item['asr_label'] for item in same_periods[name]]
                merge_dict[name]['duration']=[item['duration'] for item in same_periods[name]]
            else:
                block_id=1
                temp_len=0
                for id,item in enumerate(same_periods[name]):
                    #总时长不超过30s or 超过30s，但最后一段小于1s
                    if (temp_len+item['duration'])<=max_len or ((len(same_periods[name])-id)==1 and (item['duration']<1)):
                        merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                        merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                        merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                        temp_len=temp_len+item['duration']+insert_len
                    else:
                        block_id=(block_id+1) if id>0 else block_id
                        merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                        merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                        merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                        temp_len=item['duration']+insert_len
        #remove durtions < 1s after block segmentation 
        names=list(merge_dict.keys())
        for name in names:
            if sum(merge_dict[name]['duration']) and sum(merge_dict[name]['duration'])<1:
                merge_dict.pop(name)
        
    # insert_segment=generate_silence()#merge slince
    insert_segment=generate_car_noise()#merge noise
    fp=open(merge_asr_path,'w')
    for wav_name,item in tqdm(merge_dict.items()):
        merge_audio=concatenate_audios_with_insert(wav_dir,item['wav'],insert_segment)
        if not os.path.exists(merge_wav_dir):os.mkdir(merge_wav_dir)
        merge_audio.export(os.path.join(merge_wav_dir,(wav_name+'.wav')), format="wav")
        fp.write(wav_name+'.wav'+' '+'，'.join(item['asr']).strip()+'\n')
        fp.flush()


#datalist plus   #只用短音频做拼接 source[0]
def merge_asr_audio_with_duration_1(json_path,wav_dir,merge_wav_dir,merge_asr_path,min_len=0,max_len=30,insert_len=1,noise=False):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    merge_dict=defaultdict(lambda:{'wav':[],'asr':[],'duration':[]})

    for source in list(data.keys())[:1]:
        same_periods=defaultdict(list)
        for item in data[source]:
            key=item['audio_name'].split('.')[0]
            same_periods[key].append(item)

        len_same_periods=defaultdict(float)
        for k,v in same_periods.items():
            len_same_periods[k]=sum([subitem['duration'] for subitem in v])+(len(same_periods)-1)*insert_len

        for name,durations in len_same_periods.items():
            if durations<min_len:continue
            elif min_len<=durations<=max_len : 
                merge_dict[name]['wav']=[item['audio_name'] for item in same_periods[name]]
                merge_dict[name]['asr']=[item['asr_label'] for item in same_periods[name]]
                merge_dict[name]['duration']=[item['duration'] for item in same_periods[name]]
            else:
                block_id=1
                temp_len=0
                for id,item in enumerate(same_periods[name]):
                    if (temp_len+item['duration'])<=max_len or ((len(same_periods[name])-id)==1 and (item['duration']<1)):
                        merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                        merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                        merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                        temp_len=temp_len+item['duration']+insert_len
                    else:
                        block_id=(block_id+1) if id>0 else block_id
                        merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                        merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                        merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                        temp_len=item['duration']+insert_len
        #remove durtions < 1s after block segmentation 
        names=list(merge_dict.keys())
        for name in names:
            if sum(merge_dict[name]['duration']) and sum(merge_dict[name]['duration'])<1:
                merge_dict.pop(name)
        #source 1
        break
        
    if noise:
        insert_segment=generate_car_noise()#merge noise
    else:
        insert_segment=generate_silence()
    fp=open(merge_asr_path,'w')
    for wav_name,item in tqdm(merge_dict.items()):
        merge_audio=concatenate_audios_with_insert(wav_dir,item['wav'],insert_segment)
        if not os.path.exists(merge_wav_dir):os.mkdir(merge_wav_dir)
        merge_audio.export(os.path.join(merge_wav_dir,(wav_name+'.wav')), format="wav")
        fp.write(wav_name+'.wav'+' '+' '.join(item['asr']).strip()+'\n')
        fp.flush()


#生成label.txt
def merge_asr_audio_with_duration_without_source(json_path,wav_dir,merge_wav_dir,merge_json_path,min_len=1,max_len=30,insert_len=1,noise=False,insert_flag=True):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    merge_dict=defaultdict(lambda:{'wav':[],'asr':[],'duration':[]})

    same_periods=defaultdict(list)
    for item in data:
        key=item['audio_name'].split('.')[0]
        same_periods[key].append(item)

    len_same_periods=defaultdict(float)
    for k,v in same_periods.items():
        len_same_periods[k]=sum([subitem['duration'] for subitem in v])+(len(same_periods)-1)*insert_len

    for name,durations in tqdm(len_same_periods.items()):
        if durations<min_len:continue
        elif min_len<=durations<=max_len : 
            merge_dict[name]['wav']=[item['audio_name'] for item in same_periods[name]]
            merge_dict[name]['asr']=[item['asr_label'] for item in same_periods[name]]
            merge_dict[name]['duration']=[item['duration'] for item in same_periods[name]]
        else:
            block_id=1
            temp_len=0
            for id,item in enumerate(same_periods[name]):
                if (temp_len+item['duration'])<=max_len or ((len(same_periods[name])-id)==1 and (item['duration']<1)):
                    merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                    merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                    merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                    temp_len=temp_len+item['duration']+insert_len
                else:
                    block_id=(block_id+1) if id>0 else block_id
                    merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                    merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                    merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                    temp_len=item['duration']+insert_len
        #remove durtions < 1s after block segmentation 
        names=list(merge_dict.keys())
        for name in names:
            if sum(merge_dict[name]['duration']) and sum(merge_dict[name]['duration'])<1:
                merge_dict.pop(name)

    if noise:
        insert_segment=generate_car_noise()#merge noise
    else:
        insert_segment=generate_silence()#merge silence

    fp=open(merge_json_path,'w')
    json_data=[]
    for wav_name,item in tqdm(merge_dict.items()):
        wav_name=wav_name+'.wav' if not wav_name.endswith('.wav') else wav_name
        # merge_audio=concatenate_audios_with_insert(wav_dir,item['wav'],insert_segment,insert_flag)
        # if not os.path.exists(merge_wav_dir):os.mkdir(merge_wav_dir)
        # merge_audio.export(os.path.join(merge_wav_dir,(wav_name)), format="wav")
        # fp.write(wav_name+'.wav'+'\t'+' '.join(item['asr']).strip()+'\n')
        # fp.flush()
        temp={'audio_name':wav_name,'path':os.path.join(merge_wav_dir,wav_name),'prompt':asr_question,'asr_label':' '.join(item['asr']),'duration':sum(item['duration'])}
        json_data.append(temp)
    json.dump(json_data,fp,indent=2,ensure_ascii=False)



#拼接同一个行程中的音频，且丢弃拼接后小于1s音频，保证长度为1-30s
def merge_vad_in_group(merge_dict,merge_dict_group_keys,wav_dir,merge_wav_dir,insert_segment,insert_flag,insert_len,output_queue):
    temp_list=[]
    for key in merge_dict_group_keys:
        wav_name=key if key.endswith('.wav') else (key+'.wav')
        item=merge_dict[key]
        merge_audio=concatenate_audios_with_insert(wav_dir,item['wav'],insert_segment,insert_flag)
        if not os.path.exists(merge_wav_dir):os.mkdir(merge_wav_dir)
        merge_audio.export(os.path.join(merge_wav_dir,(wav_name)), format="wav")
        temp_dict={'audio_name':wav_name,'path':os.path.join(merge_wav_dir,wav_name),'prompt':asr_question,'asr_label':' '.join(item['asr']),'duration':sum(item['duration'])+insert_len*(len(item['duration'])-1)}
        temp_list.append(temp_dict)
    output_queue.put(temp_list)

# 创建一个全局函数，返回默认字典
def default_audio_data():
    return {'wav': [], 'asr': [], 'duration': []}

#多进程拼接
def merge_asr_audio_with_duration_without_source_mul(json_path,wav_dir,merge_wav_dir,merge_asr_path,min_len=1,max_len=30,insert_len=1,noise=False,insert_flag=True):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    # merge_dict=defaultdict(lambda:{'wav':[],'asr':[],'duration':[]})
    merge_dict=defaultdict(default_audio_data)

    same_periods=defaultdict(list)
    for item in data:
        key=item['audio_name'].split('.')[0]
        same_periods[key].append(item)

    len_same_periods=defaultdict(float)
    for k,v in same_periods.items():
        len_same_periods[k]=sum([subitem['duration'] for subitem in v])+(len(same_periods)-1)*insert_len

    for name,durations in tqdm(len_same_periods.items(),desc='Merge dictionary processing'):
        if durations<min_len:continue
        elif min_len<=durations<=max_len : 
            merge_dict[name]['wav']=[item['audio_name'] for item in same_periods[name]]
            merge_dict[name]['asr']=[item['asr_label'] for item in same_periods[name]]
            merge_dict[name]['duration']=[item['duration'] for item in same_periods[name]]
        else:
            block_id=1
            temp_len=0
            for id,item in enumerate(same_periods[name]):
                if (temp_len+item['duration'])<=max_len or ((len(same_periods[name])-id)==1 and (item['duration']<1)):
                    merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                    merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                    merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                    temp_len=temp_len+item['duration']+insert_len
                else:
                    block_id=(block_id+1) if id>0 else block_id
                    merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                    merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                    merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                    temp_len=item['duration']+insert_len
        #remove durtions < 1s after block segmentation 
        names=list(merge_dict.keys())
        for name in names:
            if sum(merge_dict[name]['duration']) and sum(merge_dict[name]['duration'])<1:
                merge_dict.pop(name)

    if noise:
        insert_segment=generate_car_noise()#merge noise
    else:
        insert_segment=generate_silence()#merge silence

    new_data=[]
    output_queue=Manager().Queue()
    merge_dict_keys=list(merge_dict.keys())
    with ProcessPoolExecutor(max_workers=40) as executor:
        group_size=round(len(merge_dict_keys)//executor._max_workers)
        merge_dict_key_groups=[merge_dict_keys[i:i+group_size] for i in range(0,len(merge_dict_keys),group_size)]
        futures=[executor.submit(merge_vad_in_group,merge_dict,group,wav_dir,merge_wav_dir,insert_segment,insert_flag,insert_len,output_queue) for group in merge_dict_key_groups]
        for future in tqdm(as_completed(futures),desc='Chunk processing'):
            future.result()
        while not output_queue.empty():
            result=output_queue.get()
            new_data+=result

    with open(merge_asr_path,'w',encoding='utf-8') as fp:
        json.dump(new_data,fp,indent=2,ensure_ascii=False)





#online_asr_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_result1.txt'
def merge_asrcor_audio_with_duration_without_source(asr_json_path,online_asr_path,wav_dir,merge_wav_dir,merge_asr_path,merge_asrcor_path,min_len=1,max_len=30,insert_len=1,noise=False):
    with open(asr_json_path,'r',encoding='utf-8') as f:
        data=json.load(f)

    online_asr_dict={}
    with open(online_asr_path,'r',encoding='utf-8') as f:
        for line in f:
            items=line.split()
            name=items[0].strip()
            online_asr=items[1].strip() if len(items)>1 else ''
            online_asr_dict[name]=online_asr

    merge_dict=defaultdict(lambda:{'wav':[],'asr':[],'duration':[],'online_asr':[]})

    same_periods=defaultdict(list)
    for item in data:
        item['online_asr']=online_asr_dict[item['audio_name']]
        key=item['audio_name'].split('.')[0]
        same_periods[key].append(item)

    len_same_periods=defaultdict(float)
    for k,v in same_periods.items():
        len_same_periods[k]=sum([subitem['duration'] for subitem in v])+(len(same_periods)-1)*insert_len

    for name,durations in len_same_periods.items():
        if durations<min_len:continue
        elif min_len<=durations<=max_len : 
            merge_dict[name]['wav']=[item['audio_name'] for item in same_periods[name]]
            merge_dict[name]['asr']=[item['asr_label'] for item in same_periods[name]]
            merge_dict[name]['duration']=[item['duration'] for item in same_periods[name]]
            merge_dict[name]['online_asr']=[item['online_asr'] for item in same_periods[name]]
        else:
            block_id=1
            temp_len=0
            for id,item in enumerate(same_periods[name]):
                if (temp_len+item['duration'])<=max_len or ((len(same_periods[name])-id)==1 and (item['duration']<1)):
                    merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                    merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                    merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                    merge_dict[name+f'_{block_id}']['online_asr'].append(item['online_asr'])
                    temp_len=temp_len+item['duration']+insert_len
                else:
                    block_id=(block_id+1) if id>0 else block_id
                    merge_dict[name+f'_{block_id}']['wav'].append(item['audio_name'])
                    merge_dict[name+f'_{block_id}']['asr'].append(item['asr_label'])
                    merge_dict[name+f'_{block_id}']['duration'].append(item['duration'])
                    merge_dict[name+f'_{block_id}']['online_asr'].append(item['online_asr'])
                    temp_len=item['duration']+insert_len
        #remove durtions < 1s after block segmentation 
        names=list(merge_dict.keys())
        for name in names:
            if sum(merge_dict[name]['duration']) and sum(merge_dict[name]['duration'])<1:
                merge_dict.pop(name)

    if noise:
        insert_segment=generate_car_noise() #merge noise
    else:
        insert_segment=generate_silence()

    fp=open(merge_asr_path,'w')
    fp1=open(merge_asrcor_path,'w')
    for wav_name,item in tqdm(merge_dict.items()):
        merge_audio=concatenate_audios_with_insert(wav_dir,item['wav'],insert_segment)
        if not os.path.exists(merge_wav_dir):os.mkdir(merge_wav_dir)
        merge_audio.export(os.path.join(merge_wav_dir,(wav_name+'.wav')), format="wav")
        fp.write(wav_name+'.wav'+' '+' '.join(item['asr']).strip()+'\n')
        fp.flush()
        fp1.write(wav_name+'.wav'+' '+' '.join(item['online_asr']).strip()+'\n')
        fp1.flush()


def merge_asr(wav_asr_path,merge_asr_path):
    fp=open(wav_asr_path,'r',encoding='utf-8')
    wav_asr=[[line.strip().split()[0],(line.strip().split()[1] if len(line.strip().split())>=2 else '')] for line in fp]
    merge_dict=defaultdict(lambda:{'wav':[],'asr':[]})
    for item in wav_asr:
        wav=item[0]
        asr=item[1]
        merge_dict[wav.split('.')[0]]['wav'].append(wav)
        merge_dict[wav.split('.')[0]]['asr']+=[asr+' ']
    fp=open(merge_asr_path,'w')
    for wav_name,item in tqdm(merge_dict.items()):
        fp.write(wav_name+'.wav'+' '+''.join(item['asr']).strip()+'\n')
        fp.flush()


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



#label.txt to json
def qa_pair_asr_with_duration_without_source(audio_dir,asr_label_path,output_path):
    with open(asr_label_path,'r',encoding='utf-8') as f:
        label_list=[{line.split('\t',maxsplit=1)[0].strip():line.split('\t',maxsplit=1)[1].strip()} for line in f]

    temp=[{'audio_name':name,'path':os.path.join(audio_dir,name),'prompt':asr_question,'asr_label':asr,'duration':librosa.get_duration(path=os.path.join(audio_dir,name))} for item in tqdm(label_list) for name,asr in item.items()]

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(temp,f,ensure_ascii=False,indent=2)


if __name__=='__main__':
    # wav_asr_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_rg_result1.txt'
    # wav_dir='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/nl0/all_wav'
    # merge_wav_dir='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_without_insert'
    # merge_asr_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_without_insert.txt'
    
    # merge_asr_audio_with_duration(wav_asr_path,wav_dir,merge_wav_dir,merge_asr_path)

    
    # wav_asr_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_result1.txt'
    # merge_asr_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_oneline_asr.txt'
    # merge_asr(wav_asr_path,merge_asr_path)

    #train secret data
    # wav_asr_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_100h_label1.txt'
    # wav_dir='/nfs/dataset-411-391/guoruizheng/train_100h'
    # merge_wav_dir='/nfs/dataset-411-391/guoruizheng/train_100h_merge'
    # merge_asr_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_100h_merge_label.txt'
    
    # merge_asr_audio(wav_asr_path,wav_dir,merge_wav_dir,merge_asr_path)


    # json_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_secret_audio_asr_new.json'
    # wav_dir='/nfs/dataset-411-391/guoruizheng/train_100h'
    # merge_wav_dir='/nfs/dataset-411-391/guoruizheng/train_100h_merge_new11'
    # merge_asr_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/train_100h_merge_label_new11.txt'

    # json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_audio_asr.json'
    # wav_dir='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/nl0/all_wav'
    # merge_wav_dir='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_0_30'
    # merge_asr_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_label_0_30.txt'
    # merge_asr_audio_with_duration_1(json_path,wav_dir,merge_wav_dir,merge_asr_path)


    # asr_json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr.json'
    # online_asr_path='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_result1.txt'
    # wav_dir='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/nl0/all_wav'
    # merge_wav_dir='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_1_30'
    # merge_asr_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_asr_1_30.txt'
    # merge_asrcor_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_asrcor_1_30.txt'
    # merge_asrcor_audio_with_duration_without_source(asr_json_path,online_asr_path,wav_dir,merge_wav_dir,merge_asr_path,merge_asrcor_path)

    # json_path='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr.json'
    # json_path_temp='/nfs/dataset-411-391/guoruizheng/train_100h_label/temp.json'
    # json_path_new='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr1.json'
    # remove_source(json_path,json_path_temp)
    # reformat_json_prompt(json_path_temp,json_path_new)

    json_path='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr1.json'
    wav_dir='/nfs/dataset-411-391/guoruizheng/train_100h'
    merge_wav_dir='/nfs/s3_k80_dataset/guoruizheng/train_100h/Audios_train_100h_merge_without_insertion_1_30'
    merge_json_path='/nfs/s3_k80_dataset/guoruizheng/train_100h/train_100h_merge_label/train_100h_merge_without_insertion_1_30_label.json'
    merge_asr_audio_with_duration_without_source(json_path,wav_dir,merge_wav_dir,merge_json_path,insert_len=0,insert_flag=False)


    # audio_dir='/nfs/dataset-411-391/guoruizheng/Audios_train_100h_merge_without_insertion_1_30'
    # asr_label_path='/nfs/dataset-411-391/guoruizheng/train_100h_merge_label/train_100h_merge_without_insertion_1_30_label.txt'
    # output_path='/nfs/dataset-411-391/guoruizheng/train_100h_merge_label/train_100h_merge_without_insertion_1_30_label_48499.json'
    # qa_pair_asr_with_duration_without_source(audio_dir,asr_label_path,output_path)
