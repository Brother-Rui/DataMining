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



prompt_asr_corretion="""<|audio_bos|><|AUDIO|><|audio_eos|>请根据给定的音频文件对以下文本 <text> 进行纠错。你的目标是对 <text> 进行必要的修改，以确保其准确反映音频内容。

要求: 
1.如果 <text> 已经与音频内容一致，则无需进行修改，输出保持不变。
2.请直接输出最终的 ASR 文本，不要随意添加任何额外的说明, 不要包含任何标点符号, 如："纠正就的文本是"。
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
prompt_asr="""<|audio_bos|><|AUDIO|><|audio_eos|>根据音频中的语音内容准确生成文本, 不要包含任何标点符号，直接输出文字内容:"""

promopt_asrcor="""<|audio_bos|><|AUDIO|><|audio_eos|>
请根据音频中的语音内容对以下文本 <text> 进行纠错。你的目标是对 <text> 进行必要的修改，以确保其准确反映音频内容。
要求: 
1.如果 <text> 已经与音频内容一致，则无需进行修改，输出保持不变。
2.请直接输出最终的 ASR 文本，不要随意添加任何额外的说明, 如："纠正就的文本是"。
3.如果音频为空或无任何内容, 则输出' '。
<text>: {text}
"""

device=torch.device(f'cuda:0')

def asr_correction(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    with open(args.label_path,'r',encoding='utf-8') as f:
        label_data=json.load(f)
    with open(args.online_asr_path,'r',encoding='utf-8') as f:
        asr_data=json.load(f)
    sr=processor.feature_extractor.sampling_rate
    # if not os.path.exists(args.result_path):
    fp=open(args.result_path,'w')

    for index,item in enumerate(label_data):

        waveform,_=librosa.load(item['path'],sr=sr)

        inputs=processor(text=prompt_asr.format(text=asr_data[index]['label']),audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result=item['name']+'\t'+response[0]
        # print(response)
        # print(type(response),len(response))
        fp.write(result+'\n')
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


def asr(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    with open(args.label_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    sr=processor.feature_extractor.sampling_rate
    # if not os.path.exists(args.result_path):
    fp=open(args.result_path,'w')

    for item in data:

        waveform,_=librosa.load(item['path'],sr=sr)

        inputs=processor(text=prompt_asr,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result=item['name']+'\t'+response[0]
        # print(response)
        # print(type(response),len(response))
        fp.write(result+'\n')
        fp.flush()

def test(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    sr=processor.feature_extractor.sampling_rate
    text='你是一个什么模型'
    waveform=[]
    inputs=processor(text=prompt_asr,audios=waveform,sampling_rate=sr,return_tensors='pt',padding=True)

    inputs.input_ids=inputs.input_ids.to(device)
    inputs['input_ids']=inputs['input_ids'].to(device)

    generate_ids=model.generate(**inputs,max_length=256)

    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(response[0])

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
            label_s1.write(item['name']+'\t'+item['label']+'\n')
        else:
            label_g1.write(item['name']+'\t'+item['label']+'\n')
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
                model_s1.write(name+'\t'+asr+'\n')
            else:
                model_g1.write(name+'\t'+asr+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default='/nfs/volume-242-2/dengh/Qwen2-audio/Qwen2-Audio-7B-Instruct')
    # parser.add_argument('--model_path',type=str,default='/nfs/volume-242-2/dengh/Qwen2-audio/models/Qwen2Audio-Train-secret-merge-new111-5epoch-4batch/checkpoint-4-36130')
    # parser.add_argument('--audio_dir',type=str)
    parser.add_argument('--label_path',type=str)
    parser.add_argument('--online_asr_path',type=str)
    parser.add_argument('--result_path',type=str)
    parser.add_argument('--final_result_path',type=str)
    parser.add_argument('--tag',type=str)
    # parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()

    # if torch.cuda_available()
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # asr_correction(args)
    # asr(args)
    
    if args.tag=='asr':
        asr(args)
        # extract_qwen_asr(args.result_path,args.final_result_path)
    elif args.tag=='asrcor':
        asr_correction(args)
        # extract_qwen_asr(args.result_path,args.final_result_path)
    
    # test(args)







