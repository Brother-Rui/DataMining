import os
import json
import random
import shutil


def merge_json_files(json_path_list,output_path):
    merged_data = {}
    for file_path in json_path_list:
        
        if file_path.endswith(".json"):
            file_name=os.path.basename(file_path)
            print(file_name)
            # Read the content of the JSON file
            with open(file_path, "r", encoding="utf-8") as json_file:
                json_data = json.load(json_file)

            # Add the data to the merged dictionary using the file name as the key
            merged_data[file_name[:-5]] = json_data

    # Path for the merged file
    merged_file_path = output_path

    # Write the merged JSON data to a new file
    with open(merged_file_path, "w", encoding="utf-8") as merged_file:
        json.dump(merged_data, merged_file, ensure_ascii=False, indent=2)

    print(f"Merged file has been saved as: {merged_file_path}")

if __name__=='__main__':
    wenetspeech_5h_path='/nfs/dataset-411-391/guoruizheng/wenetspeech/wenetspeech_sampled_5h.json'
    wenetspeech_5h_noise_path='/nfs/dataset-411-391/guoruizheng/wenetspeech/wenetspeech_sampled_5h_noise.json'
    xincheng_merge_sft_data_asr_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr.json'

    xincheng_merge_sft_data_asr_long_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_long.json'
    xincheng_merge_sft_data_asr_short_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_short.json'
    xincheng_merge_sft_data_asr_completed_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_completed.json'

    xincheng_merge_sft_data_asr_new1='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_merge_sft_data_asr_new1.json'

    #原始音频 去掉小于1s的音频
    xincheng_sft_data_ge1_asr_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_data_asr/xincheng_sft_data_ge1_asr.json'

    xincheng_sft_merge_data_asrcor_1_30_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/xincheng_sft_merge_data_asrcor_1_30.json'

    train_100h_g1_path='/nfs/dataset-411-391/guoruizheng/train_100h_label/train_100h_secret_audio_asr_g1.json'

    train_100h_merge_without_insertion_1_30_path='/nfs/s3_k80_dataset/guoruizheng/train_100h/train_100h_merge_label/train_100h_merge_without_insertion_1_30_label.json'


    # json_path_list=[xincheng_merge_sft_data_asr_long_path]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_merge_sft_data_asr_long.json'
    # merge_json_files(json_path_list,output_path)

    # json_path_list=[xincheng_merge_sft_data_asr_short_path]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_merge_sft_data_asr_short.json'
    # merge_json_files(json_path_list,output_path)

    # json_path_list=[xincheng_merge_sft_data_asr_completed_path]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_merge_sft_data_asr_completed.json'
    # merge_json_files(json_path_list,output_path)

    # json_path_list=[xincheng_sft_data_ge1_asr_path]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_sft_data_ge1_asr.json'
    # merge_json_files(json_path_list,output_path)


    # json_path_list=[wenetspeech_5h_path,xincheng_merge_sft_data_asr_path]
    # output_path='/nfs/dataset-411-391/guoruizheng/train_1220_without_noise.json'

    #json_path_list=[wenetspeech_5h_noise_path,wenetspeech_5h_path,xincheng_merge_sft_data_asr_path]
    # output_path='/nfs/dataset-411-391/guoruizheng/train_1220.json'

    # json_path_list=[xincheng_sft_merge_data_asrcor_1_30_path]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_sft_merge_data_asrcor_1_30_path.json'
    # merge_json_files(json_path_list,output_path)

    # json_path_list=[xincheng_merge_sft_data_asr_new1]
    # output_path='/nfs/volume-242-2/dengh/Qwen2-audio/src/data/train_data/xincheng_merge_sft_data_asr_new1.json'
    # merge_json_files(json_path_list,output_path)



    json_path_list=[train_100h_merge_without_insertion_1_30_path]
    output_path='/nfs/dataset-411-391/guoruizheng/train_data/train_100h_merge_without_insertion_1_30.json'
    merge_json_files(json_path_list,output_path)


    
