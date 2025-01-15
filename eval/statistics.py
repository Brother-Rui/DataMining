import os
import librosa
import json




def count_audio_files_and_duration(label_path):
    total_duration = 0.0
    audio_count = 0

    with open(label_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    audios_path=[item['path'] for item in data]

    for audio_path in audios_path:               
        duration = librosa.get_duration(path=audio_path)
        total_duration += duration
        audio_count += 1
 
    # 输出音频文件总数和总时长
    print(f"Total audio files: {audio_count}")
    print(f"Total duration (seconds): {total_duration}")
    print(f"Total duration (minutes): {total_duration / 60:.2f}")

# 指定文件夹路径
for cls in ['drunk','magic','violence']:
    print(cls)
    label_path = f'/nfs/dataset-411-391/guoruizheng/{cls}/label.json'
    count_audio_files_and_duration(label_path)
