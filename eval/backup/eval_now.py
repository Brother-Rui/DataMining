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


# torch.set_default_device('cuda:0')

# prompt_emotion="You need to complete an audio sentiment analysis task to identify the speaker's emotional state from the given audio.\
#                 Please answer directly from the following options: a. Normal b. Happy c. Sad d. Angry e. Afraid f. Other"
# prompt_age='Please analyse the age of the people in the audio and. Directly output the age.'
# prompt_sex='Please analyse the gender of the people in the audio. Directly output the gender of the speaker.'
# prompt_asr="""You need to complete an automatic speech recognition (ASR) task to convert the input audio data into corresponding text. Please generate text output accurately based on the speech content in the audio. Just directly output the text.
# Requirements:
# 1.The generated text must reflect the content in the audio as accurately as possible. Including grammar, spelling and punctuation should conform to standard writing rules.
# 2.The text output should be provided in normal text format without additional tags or formatting.
# 3.If the audio contains multiple sentences or paragraphs, please ensure that the text is appropriately segmented."""


prompt_emotion="你需要完成一个音频情绪分析任务,从给定的音频中识别说话者的情绪状态。请从以下选项中直接回答:a. 正常 b. 快乐 c. 悲伤 d. 愤怒 e. 害怕 f. 其他"
prompt_age='请分析音频中人物的年龄，直接输出年龄'
prompt_sex='请分析音频中人物的性别，直接输出性别'
prompt_asr="""你需要完成一个自动语音识别任务，将输入的音频数据转换成相应的文本。请根据音频中的语音内容准确生成文本输出,直接输出文本。
要求：
1.生成的文本必须尽可能准确地反映音频中的内容。包括语法、拼写和标点符号都应符合标准书写规则。
2.文本输出应以正常的文本格式提供，无需添加额外的标签或格式。
"""
prompt_rs_event='你正在处理一个音频分类任务,任务是识别并分类不同的人声事件。给定的音频听起来最像以下哪一种声音,从下列选项中选择一项直接回答。a.笑声 b.哭声 c.尖叫 d.咳嗽声 e.呻吟声 f.其他车内人声 g.非车内人声'
# prompt_rs_event='你需要完成一个音频分类任务，将给定音频分类为不同的人声事件声音,请从以下选项中选择一项直接回答,a.笑声 b.哭声 c.尖叫 d.咳嗽声 e.呻吟声 f.其他车内人声 g.非车内人声'
prompt_frs_device='你需要完成一个音频分类任务，将给定音频分类为不同的设备播放声音,请从以下选项中选择一项直接回答: a.TTS播报 b.音乐 c.有声书 d.视频 e.电话 f.其他'
prompt_frs_event='你需要完成一个音频分类任务，将给定音频分类为不同的非人声事件声音.请从以下选项中选择一项直接回答: a. 开关门b. 碰撞声c. 其他'


prompt_asr_corretion=''

device=torch.device('cuda:0')

def asr_correction(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav')]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)

        # duration=librosa.get_duration(y=waveform,sr=sr)
        # if duration<0.1: continue

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_frs_event}'},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs=processor(text=text,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        # print(generate_ids.shape)
        # print(generate_ids)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        result=[audio_file]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result.extend(response)
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()



def eval_frs_event(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav') and '_frs_event' in file]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)
        duration=librosa.get_duration(y=waveform,sr=sr)
        
        if duration<0.1: continue

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_frs_event}'},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs=processor(text=text,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        # print(generate_ids.shape)
        # print(generate_ids)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        result=[audio_file]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result.extend(response)
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()



def eval_frs_device(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav') and '_frs_device' in file]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)
        duration=librosa.get_duration(y=waveform,sr=sr)
        
        if duration<0.1: continue

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_frs_device}'},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs=processor(text=text,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        # print(generate_ids.shape)
        # print(generate_ids)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        result=[audio_file]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result.extend(response)
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()


def eval_rs_event(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav') and '_rs_event' in file]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)
        duration=librosa.get_duration(y=waveform,sr=sr)
        
        if duration<0.1: continue

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_rs_event}'},
            ]},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

        inputs=processor(text=text,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        # print(generate_ids.shape)
        # print(generate_ids)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        result=[audio_file]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result.extend(response)
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()

def eval_rs_asr(args):
    # accelerator=Accelerator()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav') and '_rs_asr' in file]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)
        duration=librosa.get_duration(y=waveform,sr=sr)
        
        if duration<0.1: continue

        conversation_asr = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_asr}'},
            ]},
        ]
        conversation_emotion = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_emotion}'},
            ]},
        ]
        conversation_age = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_age}'},
            ]},
        ]
        conversation_sex = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_sex}'},
            ]},
        ]

        conversations=[conversation_age,conversation_sex,conversation_emotion,conversation_asr]

        texts = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]

        audios = [waveform]*len(conversations)

        inputs=processor(text=texts,audios=audios,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        # print(generate_ids.shape)
        # print(generate_ids)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        result=[audio_file]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result.extend(response)
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()


#输出满足条件的音频名字
def audio_files_check(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    # model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav') and '_rs_asr' in file]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)
        duration=librosa.get_duration(y=waveform,sr=sr)
        
        if duration<0.1: continue

        fp.write(json.dumps([audio_file],ensure_ascii=False)+'\n')
        fp.flush()


def for_cut_record_file(args):
    input_path=args.audio_dir

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    dirs=[dir for dir in os.listdir(input_path) if os.path.isdir(os.path.join(input_path,dir))]

    fp=open(args.result_path,'w')

    for dir in tqdm(dirs):
        files=[file for file in os.listdir(os.path.join(input_path,dir)) if file.endswith('.wav')]
        for file in files:

            audio_file_path=os.path.join(input_path,dir,file)
            waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)

            conversation_asr = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "audio",
                    "audio_url": audio_file_path},
                    {"type": "text", "text": f'{prompt_asr}'},
                ]},
            ]

            text = processor.apply_chat_template(conversation_asr, add_generation_prompt=True, tokenize=False)

            inputs=processor(text=text,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

            inputs.input_ids=inputs.input_ids.to(device)
            inputs['input_ids']=inputs['input_ids'].to(device)

            generate_ids=model.generate(**inputs,max_length=256)

            generate_ids = generate_ids[:, inputs.input_ids.size(1):]

            result={'audio':os.path.join(dir,file)}
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result['asr_text']=response

            fp.write(json.dumps(result,ensure_ascii=False)+'\n')
            fp.flush()
    print(f'Asr texts saved in {args.result_path}')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default='/nfs/ofs-llm-ssd/dengh/Qwen2-audio/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--audio_dir',type=str)
    parser.add_argument('--result_path',type=str)
    parser.add_argument('--tag',type=str)
    args=parser.parse_args()

    if args.tag=='rs_asr':
        eval_rs_asr(args)
    elif args.tag=='rs_event':
        eval_rs_event(args)
    elif args.tag=='frs_event':
        eval_frs_event(args)
    elif args.tag=='frs_device':
        eval_frs_device(args)
    else:
        print('Error: without tag argument')

    # audio_files_check(args)
    # for_cut_record_file(args)

