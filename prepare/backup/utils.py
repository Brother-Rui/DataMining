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

def merge(input1,input2,output):
    with jsonlines.open(input1,'r') as f:
        data1=[item for item in f]
    with jsonlines.open(input2,'r') as f:
        data2=[item for item in f]
    print(len(data1),len(data2))
    # assert(len(data1)==len(data2),f'{len(data1)},{len(data2)}')
    for index in range(len(data1)):
        data1[index].extend(data2[index])
    print(len(data1),data1[0])
    with open(output,'w',encoding='utf-8') as f:
        json.dump(data1,f,ensure_ascii=False,indent=2)


def extract_emotion(item):
    emotion=''
    if '正常' in item:
        emotion='normal'
    elif '开心' in item:
        emotion='happy'
    elif '难过' in item:
        emotion='sad'
    elif '愤怒' in item:
        emotion='angry'
    elif '害怕' in item:
        emotion='scare'
    else:emotion='others'
    return emotion

def extract_gender(item):
    gender=''
    if '男' in item:
        gender='male'
    elif '女' in item:
        gender='female'
    else:
        gender='unsure'
    return gender

#从模型输出提取答案
def postprocess(input,output):
    with open(input,'r',encoding='utf-8') as f:
        data=json.load(f)
    new_data=[]
    for item in tqdm(data):
        temp={}
        temp['audio_name']=item[0]
        temp['sex']=extract_gender(item[2])
        temp['emotion']=extract_emotion(item[3])
        temp['age']=item[1]
        temp['asr_text']=item[4]
        new_data.append(temp)
    with open(output,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)

#对rs_asr的sex and emotion进行acc打分
def acc_score(res_path,ans_path):
    with open(res_path,'r',encoding='utf-8') as f:
        res=json.load(f)
    with open(ans_path,'r',encoding='utf-8') as f:
        ans=json.load(f)
    sex_total,sex_right=0,0
    emotion_total,emotion_right=0,0
    sex_right_dict=defaultdict(int)
    sex_total_dict=defaultdict(int)
    emotion_right_dict=defaultdict(int)
    emotion_total_dict=defaultdict(int)

    print(len(res),len(ans))
    for item1 in tqdm(res):
        for item2 in ans:
            if item1['audio_name']==item2['audio_name']:
                sex_total+=1
                emotion_total+=1
                sex_total_dict[item1['sex']]+=1
                emotion_total_dict[item1['emotion']]+=1
                if item1['emotion']==item2['emotion']:
                    emotion_right+=1
                    emotion_right_dict[item1['emotion']]+=1
                if item1['sex']==item2['sex']:
                    sex_right+=1
                    sex_right_dict[item1['sex']]+=1
                break
    sex_acc=round(sex_right/sex_total,3)
    emotion_acc=round(emotion_right/emotion_total,3)

    print(f'Sec Accuracy:{sex_acc}, right:{sex_right} {sex_right_dict}, total:{sex_total} {sex_total_dict}')
    print(f'Emotion Accuracy{emotion_acc}, right:{emotion_right} {emotion_right_dict}, total:{emotion_total} {emotion_total_dict}')
    return sex_acc,emotion_acc


#frs_event
def extract_frs_event(item):
    options_count={'tts':0,'music':0,'audiobook':0,'video':0,'phone':0,'others':0}
    if 'tts'.upper() in item or 'tts' in item:
        options_count['tts']+=1
    if '音乐' in item:
        options_count['music']+=1
    if '有声书' in item:
        options_count['audiobook'] +=1
    if '视频' in item:
        options_count['video']+=1
    if '电话' in item:
        options_count['phone']+=1
    if '其他' in item:
        options_count['others']+=1
    max_count=max(options_count.values())
    flag=0
    max_option=''
    for option,count in options_count.items():
        if count==max_count:
            max_option=option
            flag+=1
        if flag==6:
            max_option='others'
    return max_option


def frs_event_postprocess(input,output):
    with jsonlines.open(input) as f:
        data=[item for item in f]
    new_data=[]
    for item in data:
        if 'frs_event' in item[0]:
            temp={}
            temp['audio_name']=item[0]
            temp['content']=extract_frs_event(item[1])
            new_data.append(temp)
    with open(output,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)
    

def frs_event_acc(res_path,ans_path):
    with open(res_path,'r',encoding='utf-8') as f:
        res=json.load(f)
    with open(ans_path,'r',encoding='utf-8') as f:
        ans=json.load(f)

    total,right=0,0
    right_dict=defaultdict(int)
    total_dict=defaultdict(int)

    print(len(res),len(ans))
    for item1 in tqdm(res):
        for item2 in ans:
            if item1['audio_name']==item2['audio_name']:
                total+=1
                total_dict[item1['content']]+=1
                if item1['content']==item2['content']:
                    right+=1
                    right_dict[item1['content']]+=1
                break
    acc=round(right/total,3)

    print(f'Accuracy:{acc}, right:{right} {right_dict}, total:{total} {total_dict}')
    return acc

#frs_device
def extract_frs_device(item):
    cls=''
    if '开门' in item:
        cls='door'
    elif '碰撞' in item:
        cls='colision'
    elif '其他' in item:
        cls='others'
    else:
        raise ValueError(f'cls:{item}')
    return cls        


def frs_device_postprocess(input,output):
    with jsonlines.open(input) as f:
        data=[item for item in f]
    new_data=[]
    for item in data:
        if 'frs_device' in item[0]:
            temp={}
            temp['audio_name']=item[0]
            temp['content']=extract_frs_event(item[1])
            new_data.append(temp)
    with open(output,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)
    

def frs_device_acc(res_path,ans_path):
    with open(res_path,'r',encoding='utf-8') as f:
        res=json.load(f)
    with open(ans_path,'r',encoding='utf-8') as f:
        ans=json.load(f)

    total,right=0,0
    right_dict=defaultdict(int)
    total_dict=defaultdict(int)

    print(len(res),len(ans))
    for item1 in tqdm(res):
        for item2 in ans:
            if item1['audio_name']==item2['audio_name']:
                total+=1
                total_dict[item1['content']]+=1
                if item1['content']==item2['content']:
                    right+=1
                    right_dict[item1['content']]+=1
                break
    acc=round(right/total,3)
    print(f'Accuracy: {acc}')
    print(f'right:{right} {dict(right_dict)}')
    print(f'total:{total} {dict(total_dict)}')
    return acc


#rs_event
def extract_rs_event(item):
    options_count={'laughter':0,'cry':0,'scream':0,'cough':0,'moan':0,'others':0,'noncar':0}
    if '笑声' in item:
        options_count['laughter']+=1
    if '哭声' in item:
        options_count['cry']+=1
    if '尖叫' in item:
        options_count['scream'] +=1
    if '咳嗽声' in item:
        options_count['cough']+=1
    if '呻吟声' in item:
        options_count['moan']+=1
    if '其他' in item:
        options_count['others']+=1
    if '非车内人声' in item:
        options_count['noncar']+=1
    max_count=max(options_count.values())
    flag=0
    max_option=''
    for option,count in options_count.items():
        if count==max_count:
            max_option=option
            flag+=1
        if flag==6:
            max_option='others'
    return max_option


def rs_event_postprocess(input,output):
    with jsonlines.open(input) as f:
        data=[item for item in f]
    new_data=[]
    for item in data:
        if 'rs_event' in item[0]:
            temp={}
            temp['audio_name']=item[0]
            temp['content']=extract_rs_event(item[1])
            new_data.append(temp)
    with open(output,'w',encoding='utf-8') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=2)
    

def rs_event_acc(res_path,ans_path):
    with open(res_path,'r',encoding='utf-8') as f:
        res=json.load(f)
    with open(ans_path,'r',encoding='utf-8') as f:
        ans=json.load(f)

    total,right=0,0
    right_dict=defaultdict(int)
    total_dict=defaultdict(int)

    print(len(res),len(ans))
    for item1 in tqdm(res):
        for item2 in ans:
            if item1['audio_name']==item2['audio_name']:
                total+=1
                total_dict[item1['content']]+=1
                if item1['content']==item2['content']:
                    right+=1
                    right_dict[item1['content']]+=1
                break
    acc=round(right/total,3)
    print(f'Accuracy: {acc}')
    print(f'right:{right} {dict(right_dict)}')
    print(f'total:{total} {dict(total_dict)}')
    return acc



# def calculate_cer(preds, labels):
#     """
#     计算预测文本与目标文本之间的 CER（字符错误率）

#     参数:
#         preds: 模型预测输出的列表，每个元素是一个字符串
#         labels: 目标标签的列表，每个元素是一个字符串

#     返回:
#         cer: 字符错误率
#     """
#     total_distance = 0
#     total_chars = 0

#     # 遍历每一个预测和对应的标签
#     for pred, label in zip(preds, labels):
#         # 计算编辑距离（Levenshtein 距离）
#         distance = editdistance.eval(pred, label)
#         total_distance += distance
#         total_chars += len(label)  # 目标字符串的字符总数

#     # CER = 总编辑距离 / 目标文本的总字符数
#     cer = total_distance / total_chars
#     return cer

#jsonfiles
def count_secret_audio_files_and_duration(json_files,output_file,output_pic):
    with jsonlines.open(json_files,'r') as f:
        data =[obj for obj in f]

    audios_path=[item['wav'] for item in data]
    durations=[]
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



#from json_files
def count_audio_files_and_duration_from_json_without_source(json_path,output_file,output_pic):
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

#from json_files
def count_audio_files_and_duration_from_json(json_path,output_file,output_pic):
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


#from json_files
def count_audio_files_and_duration_from_json_without_source(json_path,output_file,output_pic):
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


#audio
def count_audio_files_and_duration(train_audio_dir,output_file,output_pic):
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


def ratio(txt_path,ratio_path):
    with open(txt_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    total=0
    ratio=dict()
    for k,v in data.items():
        total+=v
    for k,v in data.items():
        ratio[k]=v/total*100
    with open(ratio_path,'w',encoding='utf-8') as f:
        json.dump(ratio,f,ensure_ascii=False,indent=2)
    
#from dict.txt to record
def records(txt_path):
    with open(txt_path,encoding='utf-8') as f:
        data=json.load(f)
    # 计算最大值、最小值、音频个数、平均音频时长
    total_audio_count = 0
    weighted_sum_duration = 0
    min_duration = float('inf')
    max_duration = float('-inf')

    # 遍历所有区间
    for interval, count in data.items():
        # 获取区间的开始和结束时长
        start, end = map(int, interval[1:-1].split(','))
        
        # 更新最大值和最小值
        min_duration = min(min_duration, start)
        max_duration = max(max_duration, end)
        
        # 更新总音频个数和加权时长和
        total_audio_count += count
        weighted_sum_duration += count * ((start + end) / 2)

    # 计算平均音频时长
    average_duration = weighted_sum_duration / total_audio_count

    # 输出结果
    print("最小时长:", min_duration)
    print("最大时长:", max_duration)
    print("音频个数:", total_audio_count)
    print("平均音频时长:", average_duration)


#针对asr评测，没有文本输出的统计cer
# def no_answer_cer(secret_data_qwen2aduio_asr_model_answer_path,label_path,answer_path,record_g,record_b):
#     with open(secret_data_qwen2aduio_asr_model_answer_path,'r',encoding='utf-8') as f:
#         data=json.load(f)
#     new_data=[]
#     bad_data=[]
#     for item in data:
#         if item['model_answer']:
#             new_data.append(item)
#         else:
#             bad_data.append(item)

#     #分别记录有答案和没有答案输出的音频结果
#     res_good={}
#     durations_good=[]

#     res_bad={}
#     durations_bad=[]
#     for item in new_data:               
#         duration = item['duration']
#         key=f'[{int(duration)},{int(duration)+1})'
#         res_good[key]=res_good.get(key,0)+1
#         durations_good.append(duration)
#     for item in bad_data:               
#         duration = item['duration']
#         key=f'[{int(duration)},{int(duration)+1})'
#         res_bad[key]=res_bad.get(key,0)+1
#         durations_bad.append(duration)
        

#     sorted_keys=sorted(res_good.keys(),key=lambda x:int(x.split(',')[0][1:]))
#     sorted_res_good={k:res_good[k] for k in sorted_keys}

#     sorted_keys=sorted(res_bad.keys(),key=lambda x:int(x.split(',')[0][1:]))
#     sorted_res_bad={k:res_bad[k] for k in sorted_keys}

#     with open(record_g,'w',encoding='utf-8') as f:
#         json.dump(sorted_res_good,f,ensure_ascii=False,indent=2)

#     with open(record_b,'w',encoding='utf-8') as f:
#         json.dump(sorted_res_bad,f,ensure_ascii=False,indent=2)

#     num_audio_files_g = len(durations_good)
#     total_duration_g = sum(durations_good)
#     average_duration_g = total_duration_g / num_audio_files_g if num_audio_files_g > 0 else 0
#     max_duration_g = max(durations_good) if durations_good else 0
#     min_duration_g = min(durations_good) if durations_good else 0
    
#     print('With Answer')
#     print(f'音频文件总数: {num_audio_files_g}')
#     print(f'音频总时长: {total_duration_g:.2f} 秒')
#     print(f'音频平均时长: {average_duration_g:.2f} 秒')
#     print(f'最长音频时长: {max_duration_g:.2f} 秒')
#     print(f'最短音频时长: {min_duration_g:.2f} 秒')


#     num_audio_files_b = len(durations_bad)
#     total_duration_b = sum(durations_bad)
#     average_duration_b = total_duration_b / num_audio_files_b if num_audio_files_b > 0 else 0
#     max_duration_b = max(durations_bad) if durations_bad else 0
#     min_duration_b = min(durations_bad) if durations_bad else 0

#     print('Without Answer')
#     print(f'音频文件总数: {num_audio_files_b}')
#     print(f'音频总时长: {total_duration_b:.2f} 秒')
#     print(f'音频平均时长: {average_duration_b:.2f} 秒')
#     print(f'最长音频时长: {max_duration_b:.2f} 秒')
#     print(f'最短音频时长: {min_duration_b:.2f} 秒')

#     fp1=open(label_path,'w',encoding='utf-8')
#     fp2=open(answer_path,'w',encoding='utf-8') 
#     for item in new_data:
#         fp1.write(item['name']+' '+item['label']+'\n')
#         fp2.write(item['name']+' '+item['model_answer']+'\n')

def no_answer_cer(secret_data_qwen2aduio_asr_model_answer_path, label_path, answer_path, record_g, record_b):
    # Load data from the JSON file
    with open(secret_data_qwen2aduio_asr_model_answer_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Separate data into those with answers and without answers
    new_data = [item for item in data if item['model_answer']]
    bad_data = [item for item in data if not item['model_answer']]

    # Helper function to process durations and count occurrences
    def process_data(data):
        res = defaultdict(int)
        durations = []
        for item in data:
            duration = item['duration']
            key = f'[{int(duration)},{int(duration) + 1})'
            res[key] += 1
            durations.append(duration)
        return res, durations

    res_good, durations_good = process_data(new_data)
    res_bad, durations_bad = process_data(bad_data)

    # Sort the results by duration keys
    def sort_results(results):
        return {k: results[k] for k in sorted(results.keys(), key=lambda x: int(x.split(',')[0][1:]))}

    sorted_res_good = sort_results(res_good)
    sorted_res_bad = sort_results(res_bad)

    # Save sorted results to JSON files
    with open(record_g, 'w', encoding='utf-8') as f:
        json.dump(sorted_res_good, f, ensure_ascii=False, indent=2)

    with open(record_b, 'w', encoding='utf-8') as f:
        json.dump(sorted_res_bad, f, ensure_ascii=False, indent=2)

    # Helper function to calculate statistics
    def calculate_statistics(durations):
        num_audio_files = len(durations)
        total_duration = sum(durations)
        average_duration = total_duration / num_audio_files if num_audio_files > 0 else 0
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        return num_audio_files, total_duration, average_duration, max_duration, min_duration

    # Print statistics for data with answers
    num_audio_files_g, total_duration_g, average_duration_g, max_duration_g, min_duration_g = calculate_statistics(durations_good)
    print('With Answer')
    print(f'音频文件总数: {num_audio_files_g}')
    print(f'音频总时长: {total_duration_g:.2f} 秒')
    print(f'音频平均时长: {average_duration_g:.2f} 秒')
    print(f'最长音频时长: {max_duration_g:.2f} 秒')
    print(f'最短音频时长: {min_duration_g:.2f} 秒')

    # Print statistics for data without answers
    num_audio_files_b, total_duration_b, average_duration_b, max_duration_b, min_duration_b = calculate_statistics(durations_bad)
    print('Without Answer')
    print(f'音频文件总数: {num_audio_files_b}')
    print(f'音频总时长: {total_duration_b:.2f} 秒')
    print(f'音频平均时长: {average_duration_b:.2f} 秒')
    print(f'最长音频时长: {max_duration_b:.2f} 秒')
    print(f'最短音频时长: {min_duration_b:.2f} 秒')

    # Save labels and answers to files
    with open(label_path, 'w', encoding='utf-8') as fp1, open(answer_path, 'w', encoding='utf-8') as fp2:
        for item in new_data:
            fp1.write(f"{item['name']} {item['label']}\n")
            fp2.write(f"{item['name']} {item['model_answer']}\n")

#个数、占比
def count_duration(json_path):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    smaller_1=[]
    bet_1_30=[]
    larger_30=[]
    total_count=len(data)
    total_duration=0
    for item in data:
        duration=item['duration']
        if duration<1:
            smaller_1.append(duration)
        elif 1<=duration<=30:
            bet_1_30.append(duration)
        else:
            larger_30.append(duration)
        total_duration+=duration
    print(f't<1s, count: {len(smaller_1)},{len(smaller_1)/total_count}, duration: {sum(smaller_1)},{sum(smaller_1)/total_duration}')
    print(f'1<=t<=30, count: {len(bet_1_30)},{len(bet_1_30)/total_count}, duration: {sum(bet_1_30)},{sum(bet_1_30)/total_duration}')
    print(f't>30, count: {len(larger_30)},{len(larger_30)/total_count}, duration: {sum(larger_30)},{sum(larger_30)/total_duration}')
    

def merge_vad_count(json_path,output_path,insert=1):
    with open(json_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    merge_dict=defaultdict(list)
    group_duraion=defaultdict(list)
    group_count=defaultdict(int)
    total_count=len(data)
    total_duration=0
    for item in data:
        name=item['audio_name'].strip().split('.')[0]
        merge_dict[name].append(item['duration'])
        total_duration+=item['duration']
    for item in merge_dict.values():
        durations=(len(item)-1)*insert+sum(item)
        if durations<1:
            group_duraion['t<1'].append(item)
            group_count['t<1']+=1
        elif 30>=durations>=1:
            group_duraion['1<=t<=30'].append(item)
            group_count['1<=t<=30']+=1
        else:
            group_duraion['t>30'].append(item)
            group_count['t>30']+=1

    res=defaultdict(dict)
    for dur in group_duraion.keys():
        for item in group_duraion[dur]:
            for subitem in item:
                key=f'[{int(subitem)},{int(subitem)+1})'
                res[dur][key]=res[dur].get(key,0)+1

        sorted_keys=sorted(res[dur].keys(),key=lambda x:int(x.split(',')[0][1:]))
        sorted_temp={k:res[dur][k] for k in sorted_keys}
        res[dur]=sorted_temp

    # print(f't<1s, count: {group_count["t<1"]},{group_count["t<1"]/total_count}, duration: {sum(group_duraion["t<1"])},{sum(group_duraion["t<1"])/total_duration}')
    # print(f'1<=t<=30, count: {group_count["1<=t<=30"]},{group_count["1<=t<=30"]/total_count}, duration: {sum(group_duraion["1<=t<=30"])},{sum(group_duraion["1<=t<=30"])/total_duration}')
    # print(f't>30, count: {group_count["t>30"]},{group_count["t>30"]/total_count}, duration: {sum(group_duraion["t>30"])},{sum(group_duraion["t>30"])/total_duration}')
    
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(res,f,ensure_ascii=False,indent=2)


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
    # input1='/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_asr-Test/name.jsonl'
    # input2='/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_asr-Test/reponse.jsonl'
    # output='/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_asr-Test/qwen2audio_asr.jsonl'
    # # merge(input1,input2,output)

    # # postprocess('/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_asr-Test/qwen2audio_asr.jsonl',
    # #             '/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_asr-Test/qwe2audio_asr_dict.jsonl')
    # res_path='/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_asr-Test/qwe2audio_asr_dict.jsonl'
    # ans_path='/nfs/volume-242-5/dengh/data/xincheng_data_ori/xincheng_data_ori/raw_data_split.json'
    # # acc_score(res_path,ans_path)

    # # frs_device_postprocess('/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-frs_device-Test/reponse.jsonl',
    # #                       '/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-frs_device-Test/qwen2audio_frs_device.jsonl')
    # rs_event_postprocess('/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_event-Test/reponse.jsonl','/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_event-Test/qwen2audio_rs_event.jsonl')
    # rs_event_acc('/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-rs_event-Test/qwen2audio_rs_event.jsonl','/nfs/volume-242-5/dengh/data/xincheng_data_ori/xincheng_data_ori/raw_data_split.json')

    # 示例输入
    # preds = ["我","是","猪"]
    # labels = ["你","才","是","猪"]
    # labels1=["我","像","猪"]

    # # 计算 CER
    # cer = calculate_cer(preds, labels)
    # print(f"Character Error Rate (CER): {cer:.2f}")
    # cer = calculate_cer(preds, labels1)
    # print(f"Character Error Rate (CER): {cer:.2f}")

    # audio_dir='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_new1'
    # output_file_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_new1_record.txt'
    # output_pic_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_new1.png'
    # count_audio_files_and_duration(audio_dir,output_file_path,output_pic_path)
    # ratio(output_file_path,'/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_new1_ratio.txt')

    #test 
    # test_root='/nfs/dataset-411-391/yuchenyi/data_test/xingcheng_rerelabel/keyword'
    # output_root='/nfs/dataset-411-391/guoruizheng'
    # classes=['drunk','magic','violence']
    # for cls in tqdm(classes):
    #     json_files=os.path.join(test_root,cls,'data.list')
    #     output_file=os.path.join(output_root,cls,'record.txt')
    #     output_pic=os.path.join(output_root,cls,f'{cls}.png')
    #     ratio_file=os.path.join(output_root,cls,'ratio.txt')
    #     count_secret_audio_files_and_duration(json_files,output_file,output_pic)
    #     ratio(output_file,ratio_file)

    # txt_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_vad_new1_record.txt'
    # records(txt_path)

    # #secrete data 100h statistics
    # root='/nfs/dataset-411-391/guoruizheng'
    # experiment='sampled_train_100h_merge_new11_info'
    # dir_path=os.path.join(root,experiment)
    # if not os.path.exists(dir_path):os.mkdir(dir_path)
    # # audio_dir='/nfs/dataset-411-391/guoruizheng/train_100h_merge_new11'
    # json_path='/nfs/dataset-411-391/guoruizheng/train_100h_total/sampled_train_merge_new11_secret_audio_asr.json'
    # output_pic_path=os.path.join(dir_path,'sampled_secret_merge_new11_vad.png')
    # output_file_path=os.path.join(dir_path,'sampled_record.txt')
    # # count_audio_files_and_duration(audio_dir,output_file_path,output_pic_path)
    # count_audio_files_and_duration_from_json(json_path,output_file_path,output_pic_path)
    # #ratio 
    # output_ratio=os.path.join(dir_path,'sampled_ratio.txt')
    # ratio(output_file_path,output_ratio)


    # root='/nfs/dataset-411-391/guoruizheng/wenetspeech'
    # experiment='wenetspeech_150_info'
    # dir_path=os.path.join(root,experiment)
    # if not os.path.exists(dir_path):os.mkdir(dir_path)
    # # audio_dir='/nfs/dataset-411-391/guoruizheng/train_100h_merge_new11'
    # json_path='/nfs/dataset-411-391/guoruizheng/wenetspeech/wenetspeech_150_asr.json'
    # output_pic_path=os.path.join(dir_path,'wenetspeech_150_asr.png')
    # output_file_path=os.path.join(dir_path,'wenetspeech_150_asr_record.txt')
    # # count_audio_files_and_duration(audio_dir,output_file_path,output_pic_path)
    # count_audio_files_and_duration_from_json_without_source(json_path,output_file_path,output_pic_path)
    # #ratio
    # output_ratio=os.path.join(dir_path,'wenetspeech_150_asr_ratio.txt')
    # ratio(output_file_path,output_ratio)


    # json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_completed.json'
    # count_duration(json_path)
    # json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/xincheng_sft_data_asr.json'
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/record.txt'
    # merge_vad_count(json_path,output_path)
    # output_file='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/recode.txt'
    # output_pic='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/record.png'
    # count_audio_files_and_duration_from_json_without_source(json_path,output_file,output_pic)

    
    # #过滤掉noanswer的部分
    # root='/nfs/dataset-411-391/guoruizheng/results/Qwen2Audio-Train-secret-100h-merge-5epoch-4batch-Test'
    # for cls in tqdm(['drunk','magic','violence']):
    #     print(cls)
    #     file_path=os.path.join(root,cls,'secret_data_qwen2aduio_asr_model_answer.txt')
    #     label_path=os.path.join(root,cls,'label_only_answer.txt')
    #     model_answer_path=os.path.join(root,cls,'model_asr_only_asnwer.txt')
    #     record_g=os.path.join(root,cls,'record_g.txt')
    #     record_b=os.path.join(root,cls,'record_b.txt')
    #     no_answer_cer(file_path,label_path,model_answer_path,record_g,record_b)

    # input_path='/nfs/dataset-411-391/yuchenyi/data_test/xingcheng_rerelabel/biaozhu27699/part2/data.list'
    # output_path='/nfs/dataset-411-391/guoruizheng/xincheng_relabel_bench/bench/xincheng_relabel_bench.json'
    # bench_reformat(input_path,output_path)

    json_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_merge_sft_data_asr_without_insertion.json'
    output_file='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_without_insertion_info/record.txt'
    output_pic='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/merge_without_insertion_info/record.png'
    count_audio_files_and_duration_from_json(json_path,output_file,output_pic)
    



