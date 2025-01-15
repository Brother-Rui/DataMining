import os 
import json
#name path label
rg_asr_root='/nfs/dataset-411-391/yuchenyi/data_test/xingcheng_rerelabel/keyword'
small_asr_root='/nfs/dataset-411-391/yuchenyi/wenet_xingcheng8k/examples/xingcheng8k/s0/online.pt/attention_rescoring'
output_root='/nfs/dataset-411-391/guoruizheng'
def preprocess(label_root,small_asr_root,cls,output_root):
    text_data=[[line.split()[0].strip(),line.split()[1].strip()] for line in open(os.path.join(small_asr_root,cls,'text')).readlines()]
    wav_data=[[line.split()[0].strip(),line.split()[1].strip()] for line in open(os.path.join(label_root,cls,'wav.scp')).readlines()]
    total_res=[]
    for item in wav_data:
        res={}
        res['name']=item[0]
        res['path']=item[1]
        res['label']=next((item_text[1] for item_text in text_data if item_text[0]==item[0]),None)
        total_res.append(res)
    with open(os.path.join(output_root,cls,'small_asr.json'),'w') as f:
        json.dump(total_res,f,ensure_ascii=False,indent=2)

def change_for_cer(input_root,cls,output_root):
    input_path=os.path.join(input_root,cls,'label.json')
    with open(input_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    output_path=os.path.join(output_root,cls,'label_cer.json')
    fp=open(output_path,'w')
    for item in data:
        fp.write(item['name']+' '+item['label']+'\n')
        fp.flush()

def normal_format_label(input_path,output_path):
    fp=open(output_path,'w',encoding='utf-8')
    with open(input_path,'r',encoding='utf-8') as f:
        for line in f:
            name=line.strip().split()[0]
            asr=line.strip().split()[1]
            fp.write(name+' '+asr+'\n')
            fp.flush()



def extract_all_tar_files(file_path, extract_to_folder):
    # 确保目标文件夹存在
    os.makedirs(extract_to_folder, exist_ok=True)
    fp=open(file_path,'r',encoing='utf-8')
    tar_path_list=[line.strip() for line in fp]
    # 遍历文件夹中的所有文件
    for file_name in tar_path_list[:1]:
        # 打开并解压 .tar 文件
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=extract_to_folder)


                


if __name__=='__main__':
    # classes=['drunk','magic','violence']
    # label_root='/nfs/dataset-411-391/guoruizheng'
    # output_root='/nfs/dataset-411-391/guoruizheng/results'
    # for cls in classes:
    #     # preprocess(rg_asr_root,small_asr_root,cls,output_root)
    #     change_for_cer(label_root,cls,output_root)

    # extract_all_tar_files()

    normal_format_label(rg_asr_root+'/drunk/text','/nfs/dataset-411-391/guoruizheng/results/drunk/label_cer.json')
    