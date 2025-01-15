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
import jsonlines
import re



prompt_asr_corretion="""请根据给定的音频文件对以下文本 <text> 进行纠错。你的目标是对 <text> 进行必要的修改，以确保其准确反映音频内容。

要求: 
1.如果 <text> 已经与音频内容一致，则无需进行修改，输出保持不变。
2.请直接输出最终的 ASR 文本，不要随意添加任何额外的说明, 如："纠正就的文本是"。
3.如果音频为空或无任何内容, 则输出' '。

<text>: {text}
<asr>:

"""

# prompt_asr="""你需要完成一个中文的自动语音识别任务，将输入的音频数据转换成相应的文本<asr>。请根据音频中的语音内容准确生成文本输出,直接输出文本<asr>。
# 要求：
# 1.生成的文本必须尽可能准确地反映音频中的内容。包括语法、拼写和标点符号都应符合标准书写规则。
# 2.文本输出应以正常的文本格式提供，无需添加额外的标签或格式。
# 3.如果没有提供音频文件或音频无任何内容, 则输出空的ASR文本<asr>: 。

# """

#prompt里强调中文，有利于模型结果
prompt_asr="""你需要完成一个中文自动语音识别任务，将输入的音频数据转换成相应的文本。请根据音频中的语音内容准确生成文本输出,直接输出文本。
要求：
1.生成的文本必须尽可能准确地反映音频中的内容。包括语法、拼写和标点符号都应符合标准书写规则。
2.文本输出应以正常的文本格式提供，无需添加额外的标签或格式。
3.请直接输出文本，不要随意添加任何额外的说明。
"""


# device=torch.device('cuda:0')

def asr_correction(args):
    text_dict={}
    with open(args.text_path,'r',encoding='utf-8') as f:
        for obj in f:
            texts=obj.strip().split(':')
            text_dict[texts[0].strip()]=texts[1].strip()

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
            {'role': 'system', 'content': 'You are a helpful assistant for ASR correction.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_asr_corretion.format(text=text_dict[audio_file])}'},
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

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result=audio_file+' '+response[0]
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()

#20230611232750353028789883132e3fbcddda93c0b2d3380d0387caa4130.wav_afterVAD_231940_233750_45.wav
def extract_qwen_asr(input_path,output_path):
    fp=open(output_path,'w')
    with open(input_path,'r',encoding='utf-8') as f:
        for line in f:
            text=line.strip('\n')
            match=re.search(r"内容是：'(.*?)'",text)
            if not match: match=re.search(r"说的是：'(.*?)'",text)
            if match:
                texts=text.split(' ')
                text=texts[0]+' '+re.sub(r'[^\w\s]','',match.group(1))
                # print(text)
            elif ('AI语言模型' in text) or ('AI语言助手' in text) or ('音频' in text):
                texts=text.split(' ')
                text=texts[0]
            else: 
                texts=text.split(' ')
                text=texts[0]+' '+re.sub(r'[^\w\s]','',texts[1])
            fp.write(text+'\n')
    




def cer_score(pred_path,label_path):
    pass


def asr(args):
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
            {'role': 'system', 'content': 'You are a helpful assistant for ASR.'},
            {"role": "user", "content": [
                {"type": "audio",
                "audio_url": audio_file_path},
                {"type": "text", "text": f'{prompt_asr}'},
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

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result=audio_file+' '+response[0]
        # print(response)
        # print(type(response),len(response))
        fp.write(json.dumps(result,ensure_ascii=False)+'\n')
        fp.flush()

if __name__=='__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--model_path',type=str,default='/nfs/ofs-llm-ssd/dengh/Qwen2-audio/Qwen2-Audio-7B-Instruct')
    # parser.add_argument('--audio_dir',type=str)
    # parser.add_argument('--text_path',type=str,default='/nfs/volume-242-5/dengh/data/xincheng_sft_data/nfs/s3_k80_dataset/niemengxi/temp/asr_result.txt')
    # parser.add_argument('--result_path',type=str)
    # args=parser.parse_args()
    
    # asr_correction(args)
    # asr(args)

    # print(1)
    print('begin')
    path='/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-corr-Test/qwen_asr_new.txt'
    extract_qwen_asr(path,'/nfs/volume-242-2/dengh/Qwen2-audio/results/Qwen2-Audio-corr-Test/qwen_asr_new1.txt')

