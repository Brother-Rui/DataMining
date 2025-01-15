##Computation of CER
source /nfs/volume-411-10/jeremyyang/miniconda3/bin/activate funasr_1.1.6

for cls in drunk magic violence
do  
    experiment_name=Qwen2Audio-Train-100h_merge_without_insertion-5epoch-4batch-train-first_last2_attn+projector
    log_folder=/nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}
    label_path=/nfs/dataset-411-391/guoruizheng/results/${cls}/label_cer.json
    pred_path1=/nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}/secret_data_qwen2aduio_asr.txt
    
    python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_path} ${pred_path1} > /nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}/asr_cer.txt
    
    # pred_path2=/nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}/secret_data_qwen2aduio_asrcor.txt
    # python /nfs/volume-242-2/dengh/Qwen2-audio/src/eval/compute_wer.py --char=1 ${label_path} ${pred_path2} > /nfs/dataset-411-391/guoruizheng/results/${experiment_name}/${cls}/asrcor_cer.txt
done
