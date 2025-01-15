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


print(torch.get_default_device())
torch.cuda.set_device('cuda:1')
print(torch.get_default_device())
device=torch.get_default_device()
def eval(args):
    # accelerator=Accelerator()
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(args.model_path,device_map='auto')

    audio_dir=args.audio_dir

    audio_files=[file for file in os.listdir(args.audio_dir) if file.endswith('.wav') and '_rs_asr' in file]
    
    fp=open(args.result_path,'w')

    for audio_file in tqdm(audio_files[3880:]):    

        audio_file_path=os.path.join(audio_dir,audio_file)
        
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

        waveform,sr=librosa.load(audio_file_path,sr=processor.feature_extractor.sampling_rate)
        audios = [waveform]*len(conversations)

        inputs=processor(text=texts,audios=audios,sampling_rate=sr,return_tensors='pt',padding=True)

        inputs.input_ids=inputs.input_ids.to(device)
        inputs['input_ids']=inputs['input_ids'].to(device)

        generate_ids=model.generate(**inputs,max_length=256)

        print(generate_ids.shape)
        print(generate_ids)

        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        print(response)
        print(type(response),len(response))
        

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default='/nfs/ofs-llm-ssd/dengh/Qwen2-audio/Qwen2-Audio-7B-Instruct')
    parser.add_argument('--audio_dir',type=str)
    parser.add_argument('--result_path',type=str)
    args=parser.parse_args()
    eval(args)

